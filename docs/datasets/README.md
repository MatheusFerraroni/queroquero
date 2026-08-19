# Datasets

Inventário agregado e estado dos seis adapters. Relatórios, testes e manifests
não incluem textos, usernames, IDs pessoais ou URLs dos registros.

| Dataset | Fonte selecionada | Inventário confirmado | Preparação implementada |
| --- | --- | ---: | --- |
| [Adrenaline](adrenaline.md) | `conversations.zip` | 4.896.419 entradas; 76,1 GB | conversas TSV anonimizadas |
| [OuterSpace](outerspace.md) | `conversations.zip` | 3.650.595 entradas; 5,8 GB | conversas TSV anonimizadas |
| [BrWaC-CLEAN](brwac.md) | `BrWac.zip` → `data/*.txt` | 3.063.728 textos | seleção determinística no ZIP |
| [MultiWOZ-PTBR](multiwoz-ptbr.md) | 17 JSONs | 8.437 diálogos | utterances em ordem |
| [WackyWacky](wackywacky.md) | `pages.tsv` | 56,2 GB; linhas desconhecidas | streaming TSV e gate de boilerplate |
| [GigaVerbo-v2](gigaverbo.md) | `gigaverbo-v2/default/train-*.parquet` local | 372.108.576 linhas | streaming pinado e limitado |

`forum.*.zip`, `messages.zip` e `names.tsv` não são texto de treino.
`conversations_min.zip` é usado somente pelo perfil `smoke` dos adapters de
fórum.

## Regras comuns

- sequências exatas de 1.024 tokens, sem padding, em Parquet/ZSTD;
- `smoke`: 8 sequências de treino e 2 de avaliação;
- `mvp`: 256 sequências de treino e 32 de avaliação;
- fontes somente leitura, identificadas por fingerprints;
- limpeza leve, Unicode NFC, UTF-8 estrito, remoção de controles e HTML;
- deduplicação exata somente dentro de cada dataset;
- split estável por documento antes do packing e EOS entre documentos;
- shards atômicos, retomada, métricas e manifestos independentes;
- conteúdo e metadados sensíveis fora de Git, logs e relatórios.

Execute um adapter e revalide seus derivados com:

```sh
python -m queroquero.prepare run --dataset <id> --profile <smoke|mvp>
python -m queroquero.prepare validate --path <diretório-da-preparação>
```

Esta etapa não mistura datasets e não contém código de treinamento.
