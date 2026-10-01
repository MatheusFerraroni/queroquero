# MultiWOZ-PTBR

Inventário estrutural de 2026-08-18 dos 17 arquivos
`dialogues_001.json`–`dialogues_017.json`.

| Medida | Valor |
| --- | ---: |
| tamanho total | 174.167.553 bytes |
| diálogos | 8.437 |
| turnos | 113.552 |
| frames | 460.247 |
| turnos por diálogo | 2–44 |

Schema principal:

```text
dialogue: dialogue_id, services, turns
turn: speaker, turn_id, utterance, frames
frame: service, slots, actions, state opcional
```

Todos os arquivos inspecionados são UTF-8. O inventário observou 11 cópias
excedentes na projeção estrutural de diálogo e 15.109 ocorrências repetidas de
utterances; as contagens descrevem a fonte, não uma política adicional de
deduplicação.

## Adapter

O adapter exige exatamente os 17 JSONs e calcula SHA-256 de cada arquivo. Um
diálogo é uma unidade documental e todos os seus turnos permanecem juntos e em
ordem.

A projeção usa somente utterances:

```text
Usuário: <utterance>
Assistente: <utterance>
```

Frames, slots, actions, states, serviços e IDs não entram no texto. A limpeza é
aplicada por utterance. Diálogos são selecionados por ranking de hash estável,
e o cursor por arquivo e índice permite retomada.

O split treino/avaliação ocorre depois da projeção e antes do packing; portanto,
um diálogo nunca aparece nos dois splits. A deduplicação exata permanece
intradataset.

```sh
python -m queroquero.prepare run --dataset multiwoz_ptbr --profile smoke
```
