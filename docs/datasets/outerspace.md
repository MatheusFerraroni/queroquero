# OuterSpace

Inventário de 2026-08-18, feito somente com metadados e amostras estruturais
limitadas.

| Arquivo | Tamanho | Entradas | Uso |
| --- | ---: | ---: | --- |
| `forum.outerspace.com.br.zip` | 4.800.738.537 bytes | 570.652 | não selecionado |
| `conversations.zip` | 5.778.504.549 bytes | 3.650.595 | perfil `mvp` |
| `conversations_min.zip` | — | — | perfil `smoke` |

O inventário parcial histórico observou 1.406.509 TSVs sob `clear_threads/`
antes de interromper a enumeração remota. Essa contagem não substitui o
fingerprint integral feito pelo adapter.

## Adapter

O adapter lê diretamente membros `clear_threads/*.tsv`, sem extrair o ZIP.
Cada TSV é uma conversa sem cabeçalho, com três colunas: timestamp,
identificador de participante e texto. As mensagens permanecem na ordem da
fonte e o arquivo é a unidade documental.

Identificadores são mapeados apenas em memória para `Participante N`. Timestamp
e identificador original não entram na projeção textual. HTML é removido pela
limpeza comum, e registros que não respeitam as três colunas são rejeitados.

O fingerprint agrega nome, CRC e tamanhos dos membros do diretório central,
além do tamanho do arquivo. A seleção de conversas é determinística, tem budget
próprio e pode retomar pelo índice do membro sem persistir conteúdo em logs ou
manifests.

```sh
python -m queroquero.prepare run --dataset outerspace --profile smoke
```

O adapter prepara somente dados; treinamento e mistura com outros corpora ficam
fora desta etapa.
