import logging
from guardin_mind.manager import ConfigRead
from concurrent_log_handler import ConcurrentRotatingFileHandler
import asyncio
import edge_tts
import sounddevice as sd
import soundfile as sf
import re
import glob
import uuid
import os
import numpy

async def split_text_into_chunks(text, max_length=50):
    """
    Разбивает исходный текст на удобоваримые чанки для TTS,
    сохраняя целостность предложений и пытаясь не превышать max_length.
    """
    # Разбиваем на предложения с сохранением знаков препинания
    sentences = re.findall(r'[^.!?]+[.!?]?', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # убираем пустые

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Маленькие предложения лучше не отделять, добавляем в текущий чанк
        if len(sentence) < 15:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        else:
            # Если добавление предложения не превышает max_length — добавляем
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
            else:
                # Иначе сохраняем текущий чанк и начинаем новый
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

    # Добавляем последний оставшийся чанк
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def trim_tail_silence(data, samplerate, silence_threshold=1e-4, tail_duration=1):
    """
    Обрезает хвостовую тишину из аудиоданных, добавляемую edge-tts.
    silence_threshold — амплитуда ниже которой считается тишиной.
    tail_duration — максимальная длина хвоста тишины в секундах.
    """
    max_tail_samples = int(samplerate * tail_duration)

    reversed_data = data[::-1]

    if data.ndim == 1:  # моно
        non_silent_idx = numpy.argmax(numpy.abs(reversed_data) > silence_threshold)
    else:  # стерео или многоканал
        max_amplitude = numpy.max(numpy.abs(reversed_data), axis=1)
        non_silent_idx = numpy.argmax(max_amplitude > silence_threshold)

    trim_index = len(data) - min(non_silent_idx, max_tail_samples)
    return data[:trim_index]


class OrbitTTSEdgeTTS:
    """
    Класс для преобразования текста в речь с использованием edge-tts
    и асинхронного воспроизведения аудио с возможностью остановки.
    """

    def __init__(self):
        # Загрузка конфигурации из minder_config.toml
        ConfigRead(self)

        # Настройка логгера с ротацией файлов логов
        handler = ConcurrentRotatingFileHandler(
            "orbit-temp/orbit-logs/orbit.log", maxBytes=10**6, backupCount=5, encoding="utf-8"
        )
        logging.basicConfig(
            level=logging.INFO,
            handlers=[handler],
            format='%(asctime)s - %(levelname)s - %(pathname)s - %(message)s'
        )

        # Директория для временных аудиофайлов
        self.tmp_dir = "orbit-temp/orbit-tts"
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Очистка временной директории при старте
        for f in glob.glob(os.path.join(self.tmp_dir, '*')):
            try:
                os.remove(f)
            except Exception as e:
                logging.error(f"Не удалось удалить файл {f}: {e}")

        # Асинхронное событие для остановки воспроизведения
        self._stop_event = asyncio.Event()
        # Текущие задачи producer и consumer
        self._current_tasks = []
        # Очередь файлов для воспроизведения
        self._queue = None

    async def _cleanup_temp_files(self):
        """
        Удаляет все временные аудиофайлы из директории.
        """
        for f in glob.glob(os.path.join(self.tmp_dir, '*')):
            try:
                os.remove(f)
            except Exception as e:
                logging.error(f"Не удалось удалить файл {f}: {e}")

    async def stop(self):
        """
        Останавливает текущее воспроизведение и отменяет все задачи,
        очищает очередь и временные файлы.
        """
        self._stop_event.set()

        # Отмена всех текущих задач producer/consumer
        for task in self._current_tasks:
            task.cancel()
        self._current_tasks.clear()

        # Остановка звука (если идет воспроизведение)
        try:
            sd.stop()
        except Exception as e:
            logging.error(f"Ошибка при остановке звука: {e}")

        # Очистка очереди аудиофайлов и удаление файлов
        if self._queue:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item and os.path.exists(item):
                    try:
                        os.remove(item)
                    except Exception as e:
                        logging.error(f"Ошибка удаления файла {item}: {e}")

        await self._cleanup_temp_files()

    async def speak(
        self,
        user_text_to_tts: str,
        voice: str = "ru-RU-SvetlanaNeural",
        device_name: str | int | None = None,
        max_queue_size: int = 20,
        tail_duration: float | int = 0.7
    ) -> bool | None:
        """
        Асинхронно преобразует текст в речь и проигрывает её.
        Позволяет прерывать воспроизведение через self.stop().

        Параметры:
            user_text_to_tts — текст для озвучки
            voice — голос для edge-tts
            device_name — имя или id аудиоустройства для вывода
            max_queue_size — размер очереди чанков
            tail_duration — время хвостовой тишины для обрезки

        Возвращает:
            True при успешном воспроизведении,
            False при ошибках.
        """
        try:
            self._stop_event.clear()
            sentences = await split_text_into_chunks(user_text_to_tts)
            self._queue = asyncio.Queue(maxsize=max_queue_size)

            async def producer():
                """
                Генерирует аудиофайлы из чанков текста и кладет их в очередь.
                Останавливается при установке события стоп.
                """
                for text in sentences:
                    if self._stop_event.is_set():
                        break

                    file_path = os.path.join(self.tmp_dir, f"{uuid.uuid4()}.wav")

                    # Генерация аудиофайла с помощью edge_tts
                    tts = edge_tts.Communicate(text=text, voice=voice)
                    await tts.save(file_path)

                    # Удаляем хвостовую тишину из файла
                    data, fs = sf.read(file_path, dtype='float32')
                    trimmed_data = trim_tail_silence(data, fs, tail_duration=tail_duration)
                    sf.write(file_path, trimmed_data, fs)

                    await self._queue.put(file_path)

                # Маркер конца очереди
                await self._queue.put(None)

            async def consumer():
                """
                Извлекает аудиофайлы из очереди и проигрывает их.
                Останавливается при установке события стоп или достижении конца.
                """
                loop = asyncio.get_running_loop()
                while True:
                    if self._stop_event.is_set():
                        break

                    audio_file = await self._queue.get()
                    if audio_file is None:
                        break

                    data, fs = sf.read(audio_file, dtype='float32')

                    def play():
                        if self._stop_event.is_set():
                            return
                        if device_name:
                            sd.play(data, fs, device=device_name)
                        else:
                            sd.play(data, fs)
                        sd.wait()

                    # Запускаем воспроизведение в отдельном потоке, чтобы не блокировать asyncio
                    await loop.run_in_executor(None, play)

                    try:
                        os.remove(audio_file)
                    except Exception as e:
                        logging.error(f"Не удалось удалить файл {audio_file}: {e}")

            # Запускаем параллельно продюсера и консюмера
            self._current_tasks = [
                asyncio.create_task(producer()),
                asyncio.create_task(consumer())
            ]

            await asyncio.gather(*self._current_tasks)

            # Очистка временных файлов после завершения
            await self._cleanup_temp_files()

            return True

        except Exception as e:
            logging.error(f"Произошла ошибка во время перевода текста в речь: {e}")
            return False