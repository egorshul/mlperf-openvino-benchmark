# Whisper NPU Support Testing Scripts

Набор скриптов для проверки и тестирования поддержки Whisper ASR на NPU с multi-die конфигурацией.

## Требования

```bash
pip install openvino optimum[openvino] transformers torch soundfile
```

## Порядок выполнения

### Step 0: Подготовка модели

Если у вас нет OpenVINO модели Whisper:

```bash
python 00_prepare_model.py --output-dir ./models/whisper-large-v3-openvino
```

Это экспортирует `openai/whisper-large-v3` в формат OpenVINO с поддержкой KV-cache.

### Step 1: Проверка NPU устройств

```bash
python 01_check_npu_devices.py
```

Обнаруживает все доступные NPU dies и выводит их свойства.

**Ожидаемый вывод:**
```
✓ Found 4 NPU device(s): ['NPU.0', 'NPU.1', 'NPU.2', 'NPU.3']
✓ Multi-die configuration detected!
```

### Step 2: Проверка Optimum Intel с NPU

```bash
python 02_check_optimum_npu.py --model-path ./models/whisper-large-v3-openvino
```

Проверяет, работает ли `OVModelForSpeechSeq2Seq` напрямую с NPU.

**Возможные результаты:**
- ✅ SUCCESS → Можно использовать Optimum напрямую (простой путь)
- ❌ FAIL → Переходим к manual inference

### Step 3: Тестирование manual encoder-decoder

```bash
python 03_test_manual_encoder_decoder.py --model-path ./models/whisper-large-v3-openvino --device NPU.0
```

Загружает encoder и decoder отдельно через OpenVINO Core API.

**Проверяет:**
- Загрузка encoder на NPU
- Загрузка decoder на NPU
- Базовый inference для обоих

### Step 4: Тестирование multi-die распределения

```bash
python 04_test_multi_die.py --model-path ./models/whisper-large-v3-openvino --num-samples 16
```

Тестирует параллельный encoder inference на нескольких dies.

**Метрики:**
- Throughput на одном die
- Throughput при параллельном использовании всех dies
- Коэффициент масштабирования

### Step 5: Полный benchmark Whisper

```bash
# С dummy audio
python 05_full_whisper_benchmark.py --model-path ./models/whisper-large-v3-openvino --num-samples 8

# С реальным аудио
python 05_full_whisper_benchmark.py --model-path ./models/whisper-large-v3-openvino --audio test.wav
```

Полный pipeline: encoder → decoder → генерация текста.

## Структура экспортированной модели

После экспорта через `optimum-cli`:

```
whisper-large-v3-openvino/
├── encoder_model.xml          # Encoder (mel → hidden states)
├── encoder_model.bin
├── decoder_model.xml          # Decoder первого шага (без KV-cache)
├── decoder_model.bin
├── decoder_with_past_model.xml # Decoder с KV-cache (для последующих шагов)
├── decoder_with_past_model.bin
├── config.json
├── tokenizer.json
├── preprocessor_config.json
└── generation_config.json
```

## Архитектура multi-die inference

```
┌─────────────────────────────────────────────────────────────┐
│                  WhisperMultiDeviceSUT                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sample 0 ──→ NPU.0: [Encoder] ──→ [Decoder] ──→ Text      │
│  Sample 1 ──→ NPU.1: [Encoder] ──→ [Decoder] ──→ Text      │
│  Sample 2 ──→ NPU.2: [Encoder] ──→ [Decoder] ──→ Text      │
│  Sample 3 ──→ NPU.3: [Encoder] ──→ [Decoder] ──→ Text      │
│  Sample 4 ──→ NPU.0: [Encoder] ──→ [Decoder] ──→ Text      │
│  ...                                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Каждый sample обрабатывается полностью на одном die (round-robin распределение).

## Troubleshooting

### "No NPU devices found"
- Проверьте драйвер NPU
- Проверьте OpenVINO NPU plugin

### "Encoder/Decoder inference failed"
- Попробуйте конкретный die: `--device NPU.0`
- Проверьте формат модели
- Попробуйте другую версию OpenVINO

### "KV-cache input needs proper handling"
- Decoder с KV-cache требует специальной обработки
- Используйте `decoder_model.xml` вместо `decoder_with_past_model.xml` для тестирования

## Следующие шаги

После успешного прохождения всех тестов:

1. Создать `whisper_multi_device_sut.py` в `src/mlperf_openvino/core/`
2. Интегрировать в `sut_factory.py`
3. Обновить `benchmark_runner.py`
4. Запустить MLPerf benchmark
