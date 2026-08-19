# Adrenaline

Inspeção de 2026-08-18, somente de metadados e amostras limitadas.

| Arquivo | Tamanho | Entradas | Uso |
| --- | ---: | ---: | --- |
| `forum.adrenaline.com.br.zip` | 2.157.145.545 bytes | 356.799 | não selecionado |
| `conversations.zip` | 76.123.303.722 bytes | 4.896.419 | selecionado |
| `conversations_min.zip` | — | — | fixture de parser |
| `messages.zip` | — | — | não selecionado |

O ZIP de fórum contém JSONs de categorias e tópicos; as amostras não
confirmaram um corpus conversacional completo. O ZIP selecionado é grande
demais para enumeração repetida via SSHFS e seu schema ainda não foi confirmado.

## Antes da preparação

1. gerar uma vez o manifesto do diretório central no host dos dados;
2. validar, em poucas entradas, formato, encoding, colunas e unidade documental;
3. definir filtros de texto, PII, HTML, duplicação e tamanho;
4. processar por shard com cursor e budget próprios.

Não extrair o ZIP completo nem registrar valores dos documentos.
