# MultiWOZ-PTBR

Inspeção estrutural de 2026-08-18 dos 17 arquivos
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

Todos os arquivos são UTF-8. Speakers estão balanceados entre `USER` e
`SYSTEM`. Foram observadas 11 cópias excedentes na projeção estrutural de
diálogo e 15.109 ocorrências repetidas de utterances.

## Decisão

Usar todos os 17 arquivos e preservar diálogo, ordem dos turnos e arquivo de
origem. A projeção textual para pretraining será derivada e configurável.

## Antes da preparação

- decidir marcadores de speaker e exclusão de frames/slots do texto;
- definir deduplicação por diálogo e/ou utterance;
- medir tokens e escolher truncamento ou descarte;
- gerar shards retomáveis por arquivo e índice de diálogo.
