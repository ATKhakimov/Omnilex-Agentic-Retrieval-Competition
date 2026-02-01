# Omnilex Agentic Retrieval Competition Starter Repo

Official starter repo for Kaggle competiton https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/host/launch-checklist

## Quick Start

### Installation

(Tested with Ubuntu-24.04 in WSL)

```bash
# Clone the repository
git clone https://github.com/Omnilex-AI/Omnilex-Agentic-Retrieval-Competition.git
cd Omnilex-Agentic-Retrieval-Competition

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # for testing/linting

# Install package in development mode
pip install -e .
```

### Download Data

Get it from Kaggle into `data` directory

### Run Baselines

Two baseline notebooks are provided:

1. **Direct Generation** (`notebooks/01_direct_generation_baseline.ipynb`)
   - Prompts LLM to directly generate citations
   - Simple but prone to hallucination

2. **Agentic Retrieval** (`notebooks/02_agentic_retrieval_baseline.ipynb`)
   - Uses ReAct-style agent with search tools
   - Grounded in actual legal documents

Both notebooks work in VSCode and can be submitted to Kaggle.

### Validate Submission

```bash
python scripts/validate_submission.py submission.csv
```

## Data Format

See Kaggle

## Project Structure

```
├── src/omnilex/           # Core library
│   ├── citations/         # Citation parsing & normalization
│   ├── evaluation/        # Metrics & scoring
│   ├── retrieval/         # BM25 search & tools
│   └── llm/               # LLM loading & prompts
├── notebooks/             # Baseline notebooks
├── utils/                 # Data & utility scripts
├── tests/                 # Test suite
└── data/                  # Data directory
```

## Requirements

- Python >= 3.10
- llama-cpp-python (for local LLM inference)
- rank-bm25 (for keyword search)
- pandas, numpy, scikit-learn

For Kaggle submissions, you may need to (depending on your solution):

1. Upload your GGUF model as a Kaggle dataset
2. Upload pre-built indices as a Kaggle dataset
3. Package the `omnilex` library

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Contact

For public questions about the competition please use the "Discussion" tab or open an issue on this repository. For private questions reahc out to host on Kaggle or ari.jordan@omnilex.ai

# Swiss Legal Information Retrieval Challenge — Overview

## Цель соревнования
Соревнование направлено на построение **LLM-powered agentic системы юридического поиска** для швейцарского права.  
Для каждого юридического вопроса на английском языке система должна **найти и вернуть наиболее релевантные источники швейцарского права** (статьи законов, судебные решения и их рассмотрения), оформленные в виде списка канонических цитат.

Ключевая задача — **максимально точно воспроизвести набор эталонных цитат**, используемых экспертами, на скрытом тестовом наборе.

---

## Формулировка задачи
- **Вход**: юридический вопрос на английском языке  
- **Выход**: список релевантных швейцарских правовых источников  
  - формат: `citation_1;citation_2;...`
  - цитаты преимущественно на немецком (реалистичный кросс-языковой сценарий)
- **Ограничений на подход нет**: допускаются BM25, dense retrieval, reranking, agentic pipelines, graph-based retrieval, гибридные системы и др.

---

## Тип соревнования
- **Code competition (Kaggle)**
- Обязательна **офлайн-воспроизводимость**:
  - ноутбук без доступа к интернету
  - ограниченные вычислительные ресурсы
  - полный прогон ≤ **12 часов**
- Для призов требуется **ноутбук**, а не только `submission.csv`

---

## Оценка качества
- **Метрика**: Citation-level **Macro F1**
  - считается **по каждому запросу**
  - затем усредняется по всем запросам
- Метрика поощряет:
  - стабильную точность
  - корректную работу с вопросами разной сложности
- **Leaderboard**:
  - Public: 50% теста (фидбек во время соревнования)
  - Private: 50% теста (финальный рейтинг)

---

## Данные

### 1. Training set
- Основан на **LEXam (Fan et al., 2025)**
- Открытые вопросы с извлечёнными «золотыми» цитатами
- Используется для обучения и разработки
- Лицензия: **CC BY 4.0**

### 2. Validation set
- Малый набор (10 вопросов)
- Английские запросы
- **Не совпадает по распределению** с train
- Используется для sanity-check

### 3. Test set
- Только запросы (без разметки)
- 40 английских вопросов
  - 20 — public LB
  - 20 — private LB
- Существует дополнительный **HIDDEN.csv** для анти-читинг-проверок

### 4. Retrieval corpus
Поисковое пространство (закрытый словарь цитат):
- `laws_de.csv`
  - фрагменты федеральных законов (немецкий)
- `court_considerations.csv` (очень большой)
  - мотивировочные части решений Федерального суда
  - языки: немецкий / французский / итальянский
  - охват ≈ 30 лет
  - неполный, но достаточный для задачи

---

## Формат данных

### train.csv / val.csv
| column | описание |
|------|----------|
| query_id | уникальный ID |
| query | вопрос (EN) |
| gold_citations | эталонные цитаты через `;` |

### test.csv
| column | описание |
|------|----------|
| query_id | ID |
| query | вопрос (EN) |

### laws_de.csv
| column | описание |
|------|----------|
| citation | канонический идентификатор |
| text | текст закона (DE) |

### court_considerations.csv
| column | описание |
|------|----------|
| citation | канонический идентификатор |
| text | текст рассмотрения |

---

## Формат сабмишена
```csv
query_id,predicted_citations
test_0001,"Art. 11 Abs. 2 OR;BGE 139 I 2 E. 3.1"
test_0002,"5A_800/2019 E 5."
test_0003,""
````

* одна строка на `query_id`
* допустима пустая строка
* **важна точность формата** — даже мелкие расхождения считаются ошибкой

---

## Гранулярность поиска

* **Каждая строка корпуса — отдельная единица поиска**
* Даже если цитаты относятся к одному решению, разные канонические строки считаются **разными ID**
* Цитаты можно рассматривать как **opaque ID**, не обязательно их парсить

---

## Типы цитат (ориентиры)

* **Федеральные законы**

  * `Art. 1 ZGB`
  * `Art. 11 Abs. 2 OR` (параграфы важнее целых статей)
* **Решения суда**

  * Ведущие: `BGE 145 II 32 E. 3.1`
  * Неведущие: `5A_800/2019 E 2.`

---

## Практические замечания

* **Citation precision критична** — LLM-генерация часто даёт мелкие ошибки
* **Утечек нет**: тест намеренно не совпадает с train / публичными данными
* Победителей могут попросить:

  * краткое описание решения
  * подтверждение воспроизводимости

---

## Особенности домена

* Юридическая «истина» **не единственна**
* Gold citations — это:

  * качественные экспертные ориентиры
  * но с возможным шумом и субъективностью
* Фактически модель учится **предсказывать выбор аннотатора**
* Задачу можно рассматривать как:

  * few-shot transfer learning
  * noisy multi-label retrieval

---

## Ключевые вызовы

* Кросс-языковой поиск (EN → DE/FR/IT)
* Многоцитатные ответы
* Шумная разметка
* Жёсткие требования к формату
* Офлайн-ограничения и воспроизводимость

---

## Итог

Соревнование — это реалистичный сценарий **agentic legal retrieval**:

* не генерация текста,
* а точный, воспроизводимый **поиск и ранжирование правовых источников**
  в условиях юридической неоднозначности и производственных ограничений.



