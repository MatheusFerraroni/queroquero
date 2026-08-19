# Adrenaline

Inventário de 2026-08-18, feito somente com metadados e amostras estruturais
limitadas.

| Arquivo | Tamanho | Entradas | Uso |
| --- | ---: | ---: | --- |
| `forum.adrenaline.com.br.zip` | 2.157.145.545 bytes | 356.799 | não selecionado |
| `conversations.zip` | 76.123.303.722 bytes | 4.896.419 | perfil `mvp` |
| `conversations_min.zip` | — | — | perfil `smoke` |
| `messages.zip` | — | — | não selecionado |

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
python -m queroquero.prepare run --dataset adrenaline --profile smoke
```

O adapter prepara somente dados; treinamento e mistura com outros corpora ficam
fora desta etapa.
