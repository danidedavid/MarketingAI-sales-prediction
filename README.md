# Previsão de Vendas – TCC CDPRO

![tests](https://github.com/USER/REPO/actions/workflows/tests.yml/badge.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

**App e pipeline para previsão de vendas por item×loja** usando *features* temporais (calendário, `lag1`), 
clustering prévio (opcional) e modelos comparados (**Ridge**, **LightGBM**, **RandomForest**). 
Inclui **Streamlit** para predição unitária, **scripts** modulares em `src/` e **testes** com `pytest`.

> Substitua `USER/REPO` acima pelo **seu** usuário e repositório do GitHub para o badge funcionar.

---

## Visão Geral do Projeto
O objetivo é estimar a demanda mensal por par **item×loja**, com foco em rodar rápido em ambientes como Google Colab
e VSCode. O pipeline gera *features* de calendário, `lag1` por grupo, faz validação temporal (sem fuga de informação),
treina modelos, escolhe o melhor e salva:
- `outputs/best_model_pipeline.pkl` – pipeline de inferência (pré-processamento + modelo)
- `outputs/schema.json` – ordem/nomes de features e colunas categóricas esperadas
- `outputs/fallback_stats.json` – medianas e default para preencher ausentes na inferência

---

## Estrutura do Repositório
```text
project/
├── src/
│   ├── __init__.py
│   ├── data.py            # leitura eficiente, downcast, datas mensais
│   ├── features.py        # calendário, lag1, rollings (opcional)
│   ├── models.py          # construção/avaliação de modelos, split temporal
│   └── visualization.py   # gráficos básicos (matplotlib)
├── notebooks/             # exploração/EDA/treino (opcional)
├── outputs/
│   ├── best_model_pipeline.pkl
│   ├── schema.json
│   └── fallback_stats.json
├── tests/
│   ├── conftest.py
│   └── test_all.py
├── .github/workflows/tests.yml
├── streamlit_app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Requisitos e Instalação
> Recomendado **Python 3.10 ou 3.11**.

1. Crie e ative a venv (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   macOS/Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uso

### 1) App Streamlit (predição unitária)
```bash
streamlit run streamlit_app.py
```
Na **sidebar**, aponte para:
- `outputs/best_model_pipeline.pkl`
- `outputs/schema.json`
- `outputs/fallback_stats.json`

Preencha **Ano, Mês, item, store** e (opcionalmente) `lag1` (vendas do mês anterior para aquele par).

### 2) Testes
```bash
pytest -q
```

---

## Metodologia
- **Features**: calendário (`year`, `month`, `quarter`, `month_sin`, `month_cos`) + `lag1` por grupo (**item×loja**).
- **Validação temporal**: treino em meses anteriores e validação no **último mês** (ou *folds* recentes).
- **Modelagem**: comparação de **Ridge**, **RandomForest** e **LightGBM**; seleção pelo menor **RMSE**.
- **Inferência**: `schema.json` garante a **ordem de features**; `fallback_stats.json` preenche **ausentes**.

---

## Resultados
- Métricas consolidadas por *fold* e modelo (ex.: `metrics_temporal_fast.csv` nos notebooks rápidos).
- Pipeline vencedor salvo em `outputs/` pronto para carga no Streamlit ou em lotes.

---

## Deploy

### A) Streamlit Community Cloud (gratuito)
1. Suba o repositório no GitHub com a estrutura acima.
2. Acesse **https://share.streamlit.io** e conecte-se ao GitHub.
3. Crie um novo app escolhendo `USER/REPO` e defina **Main file = `streamlit_app.py`**.
4. (Opcional) Adicione segredos (chaves/URLs) em **`/.streamlit/secrets.toml`** (o arquivo já está ignorado no `.gitignore`).
5. Deploy! O Streamlit instalará `requirements.txt` automaticamente.

### B) Render (alternativa simples)
1. Crie um **Web Service** apontando para o seu repositório.
2. **Build command**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start command**:
   ```bash
   streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Defina o **Python** para 3.10/3.11 e ative cache de dependências.

### C) GitHub Actions (testes)
- Já há um workflow `tests.yml` que roda `pytest` em cada push/PR em `main`/`master`.
- Badge no topo do README (substitua `USER/REPO` pelo seu):
  ```markdown
  ![tests](https://github.com/USER/REPO/actions/workflows/tests.yml/badge.svg)
  ```

---

## Contribuição
1. Faça um fork e crie um branch: `feat/minha-feature`.
2. Rode `pytest` e garanta que os testes passam.
3. Abra PR descrevendo mudanças e motivações.

---

## Autores
- Daniela de David (coordenação e desenvolvimento)
- Colaborações via Pull Requests são bem-vindas.

---

## Licença
Este projeto é distribuído sob a **Licença MIT**. Veja [`LICENSE`](LICENSE).

---

## Agradecimentos
- Equipe/Comunidade CDPRO
- Bibliotecas: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `streamlit`, `matplotlib`.
