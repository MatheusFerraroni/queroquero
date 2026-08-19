# MultiWOZ-PTBR: exploração estrutural dos diálogos

Data da inspeção: 2026-08-18. O diretório
`$PTBR_DATASET_ROOT/multiwozptbr` está em um mount SSHFS somente leitura. Esta
nota registra apenas metadados, schema e estatísticas agregadas. Nenhum texto de
utterance, nome, ID, valor de slot ou outro conteúdo de diálogo foi persistido
no repositório.

## Inventário

Foram encontrados exatamente 17 arquivos, de `dialogues_001.json` a
`dialogues_017.json`. Todos foram lidos uma vez diretamente no mount, sem
cópia, extração ou arquivo intermediário local.

| Grupo | Arquivos | Tamanho total | Diálogos | Turnos | Frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dialogues_001`–`dialogues_016` | 16 | 169.065.763 bytes | 8.191 | 110.230 | 446.786 |
| `dialogues_017` | 1 | 5.101.790 bytes | 246 | 3.322 | 13.461 |
| **total** | **17** | **174.167.553 bytes** | **8.437** | **113.552** | **460.247** |

Dezesseis arquivos têm 512 diálogos; `dialogues_007` tem 511 e o último tem
246. Os arquivos funcionam como uma divisão física em lotes de aproximadamente
512 diálogos, e não como divisões semânticas independentes confirmadas.

## Schema observado

Todos os 17 arquivos têm raiz JSON do tipo `list`. Todos os 8.437 itens da
lista são objetos com exatamente estas chaves:

```text
dialogue_id: string
services: list
turns: list
```

Os 113.552 turns têm schema consistente:

```text
speaker: string
turn_id: string
utterance: string
frames: list
```

Não foram encontrados campos ausentes ou nulos em diálogos ou turns. Os
`turn_id` são strings curtas; seus valores não foram registrados.

Há 460.247 frames. Todos têm `service`, `slots` e `actions`; 454.208 também
têm `state`, enquanto 6.039 não têm essa chave. O objeto `state`, quando
presente, tem sempre:

```text
active_intent: ...
requested_slots: ...
slot_values: ...
```

Todos os `actions` observados são listas vazias; não foram observados objetos
de ação. Há 21.912 slots em duas formas estruturais:

| Forma | Quantidade |
| --- | ---: |
| `slot`, `value`, `start`, `exclusive_end`, `translated` | 20.391 |
| `slot`, `value`, `copy_from` | 1.521 |

O campo `value` é string na primeira forma e lista na segunda. Os offsets são
numéricos e `translated` é booleano. Os valores concretos não foram expostos.

## Estatísticas agregadas

- A quantidade de speakers é equilibrada: 56.776 `USER` e 56.776 `SYSTEM`.
- Há de 2 a 44 turns por diálogo; média 13,459 e mediana 14.
- Há de 0 a 8 frames por turn; média 4,053 e mediana 5.
- Há de 3 a 533 caracteres por utterance; média 71,827 e mediana 66.
- A leitura UTF-8 estrita funcionou nos 17 arquivos; nenhum possui BOM UTF-8.
  Foram 8.156.092 caracteres e 8.398.644 bytes UTF-8 nas utterances; a
  proporção agregada de caracteres não ASCII foi 2,974%.
- O campo `services` tem de 0 a 4 itens; média 1,710 e mediana 2. A
  distribuição de cardinalidade é 417 diálogos com zero, 2.833 com um, 3.994
  com dois, 1.168 com três e 25 com quatro serviços.
- Agregados, os frames cobrem oito serviços de domínio. Em cada um dos 8.437
  diálogos, o conjunto de `services` declarado é subconjunto do conjunto de
  serviços presentes nos frames; portanto, não se deve contar cada frame como
  uma ocorrência de diálogo ativa. Frames vazios e estados sem conteúdo
  precisam ser filtrados conforme a tarefa.

Contagem agregada do campo `service` nos frames:

| Serviço | Frames |
| --- | ---: |
| `restaurant` | 59.036 |
| `hotel` | 58.819 |
| `attraction` | 57.761 |
| `taxi` | 57.377 |
| `train` | 56.920 |
| `hospital` | 56.782 |
| `bus` | 56.776 |
| `police` | 56.776 |

Para verificar duplicação sem registrar identificadores ou textos, foi usado
somente hash SHA-256 em memória:

- os 8.437 IDs produziram 8.437 hashes distintos;
- o hash da projeção estrutural avaliada (sequência de utterances e forma dos
  frames) produziu 8.426 buckets, com excesso de 11 cópias nessa projeção;
- utterances idênticas produziram 98.443 buckets, com excesso de 15.109
  ocorrências e maior bucket de 307 ocorrências. Isso é compatível com frases
  genéricas recorrentes, mas justifica deduplicação no derivado de treino.

O hash foi descartado ao fim da inspeção e nenhum valor foi escrito em arquivo.

## Implicações para ingestão

Este é um corpus conversacional estruturado, diferente de um corpus livre de
texto. A unidade natural de preservação é um diálogo completo, mantendo a
ordem dos turns e a proveniência do arquivo de origem. Para continual
pretraining, a projeção mais simples é converter cada diálogo em um documento
textual com marcadores de speaker, mas isso deve ser uma etapa derivada e
configurável. Para tarefas de diálogo/estado, convém preservar também frames,
slots e state em um formato separado, sem misturar os campos de anotação ao
texto destinado ao modelo.

Antes de usar o corpus, a configuração deve decidir explicitamente:

1. se frames sem `state` e listas vazias de `actions` serão descartados;
2. se os marcadores `USER`/`SYSTEM` serão mantidos;
3. se slots, offsets e estados serão excluídos do texto de pretraining ou
   usados apenas em uma tarefa supervisionada;
4. se a deduplicação será por diálogo completo, por utterance ou ambas;
5. se o limite de contexto será aplicado truncando turns ou descartando o
   diálogo inteiro.

O maior diálogo observado tem 44 turns; a distribuição de tamanho de texto
deve ser medida em tokens do tokenizer escolhido antes de fixar o limite de
contexto. Caracteres e turns não são substitutos seguros para essa medida.

## Não é necessário reler tudo em cada execução

Uma leitura completa foi útil para estabelecer este inventário, mas não deve
fazer parte de cada treino. O fluxo recomendado é:

1. calcular uma identificação da versão do conjunto com caminho, tamanho e
   mtime dos 17 arquivos; usar hash completo apenas quando houver janela de
   I/O adequada;
2. gerar uma vez um manifesto local fora do Git, com arquivo de origem,
   quantidade de diálogos, turns, bytes, schema e estatísticas agregadas;
3. fazer uma amostragem determinística bounded (por exemplo, arquivos 001, 009
   e 017 e um número fixo de diálogos por arquivo) para validar schema,
   tokenização, duplicação e filtros;
4. processar cada diálogo em streaming, escrevendo shards derivados pequenos
   com `source_file`, índice ordinal do diálogo, configuração de normalização
   e contagem de tokens;
5. retomar por arquivo/índice de diálogo ou shard concluído, validando o
   fingerprint antes de reutilizar o cache;
6. no treino, consumir apenas os shards e o limite de tokens escolhido.

O manifesto, shards, caches, hashes detalhados e textos normalizados devem
ficar fora do Git. O repositório deve conter apenas este relatório, schema,
configuração e métricas agregadas.

## Comandos seguros e reproduzíveis

Para confirmar o inventário físico sem abrir os documentos:

```sh
for f in "$PTBR_DATASET_ROOT"/multiwozptbr/dialogues_*.json; do
  stat -f '%N|%z bytes|%Sm' -t '%Y-%m-%d %H:%M:%S %z' "$f"
done
```

Para uma nova verificação bounded, o script deve abrir um arquivo por vez,
contar tipos/chaves e descartar o objeto antes de passar ao próximo. A saída
deve conter somente contagens e estatísticas, nunca `dialogue_id`,
`utterance`, `slot`, `value` ou outros valores. Um esqueleto seguro é:

```python
import json
import os
from pathlib import Path

root = Path(os.environ['PTBR_DATASET_ROOT']) / 'multiwozptbr'
sample = [
    root / 'dialogues_001.json',
    root / 'dialogues_009.json',
    root / 'dialogues_017.json',
]

for path in sample:
    with path.open('r', encoding='utf-8') as stream:
        rows = json.load(stream)
    # Emitir somente len(rows), tipos, chaves e contagens agregadas.
    del rows
```

O acesso remoto deve ser mantido somente leitura. Não usar `cp`, `unzip`,
extração integral ou scripts que escrevam no diretório montado. Para produção,
é preferível executar a preparação no host que armazena os arquivos e trazer
apenas o manifesto/shards derivados necessários.

## Limitações e próxima decisão

- A inspeção confirmou a estrutura dos 17 JSONs, mas não confirmou uma divisão
  oficial train/dev/test; os nomes dos arquivos indicam apenas lotes físicos.
- A contagem de tokens depende do tokenizer/modelo e ainda não foi medida.
- A deduplicação foi medida por hash em memória; ainda não há política de
  remoção escolhida.
- Não foi feita classificação de qualidade linguística nem sanitização dos
  valores de slots, pois isso exigiria definir o uso (pretraining versus tarefa
  de diálogo).
- Antes de escolher quanto usar, faça uma amostra bounded para tokenização,
  distribuição de comprimento, impacto dos filtros e tamanho dos shards. O
  conjunto completo só deve ser materializado uma vez se a decisão de uso
  justificar esse custo.
