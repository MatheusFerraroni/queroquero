# OuterSpace: exploração inicial dos arquivos montados

Data da inspeção: 2026-08-18. O mount externo, representado nos comandos por
`$PTBR_DATASET_ROOT`, é SSHFS e somente leitura. Esta nota registra apenas
metadados, schemas e estatísticas agregadas. Nenhum ZIP foi extraído
integralmente e nenhum texto, username ou URL foi incluído aqui.

## Inventário

| Arquivo | Tamanho no mount | Entradas conhecidas | Estrutura observada |
| --- | ---: | ---: | --- |
| `outerspace/forum.outerspace.com.br.zip` | 4.800.738.537 bytes | 570.652 (570.649 arquivos JSON e 3 diretórios) | exportação hierárquica do fórum |
| `outerspace/conversations.zip` | 5.778.504.549 bytes | 3.650.595 no cabeçalho ZIP; 1.406.509 TSVs observados antes da interrupção | `clear_threads/<identificador>.tsv` na enumeração parcial |

Os tamanhos são os retornados por `stat` no mount; não representam tamanho
descompactado. As contagens totais vêm do cabeçalho ZIP; as estatísticas e
estruturas observadas usam somente o diretório central, sem extrair membros.

## `forum.outerspace.com.br.zip`

O diretório raiz contém três grupos de dados, além do diretório raiz:

- `categories.json`: um array com 8 categorias. Cada categoria tem os campos
  `id`, `subs`, `title_href` e `title_text`; cada item de `subs` tem
  `complete`, `description`, `id`, `last_update`, `title_href` e `title_text`.
- `categories_threads/`: 26 arquivos JSON, cada um associado a uma categoria e
  subcategoria.
- `threads/`: 570.621 arquivos JSON, todos com nome numérico seguido de
  `.json`.
- `config.json`: objeto pequeno com as chaves `domain`, `last_id` e `url`.

Estatísticas do diretório central:

| Grupo | Arquivos | Descompactado | Compactado | Mediana | P90 | Máximo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `categories_threads/` | 26 | 235.798.966 bytes | 36.327.941 bytes | 5.998.881 bytes | 11.186.423 bytes | 55.882.605 bytes |
| `threads/` | 570.621 | 26.461.028.887 bytes | 4.651.393.247 bytes | 13.094 bytes | 75.758 bytes | 495.048.679 bytes |
| total de arquivos | 570.649 | 26.696.849.384 bytes | 4.687.723.489 bytes | — | — | — |

Todos os arquivos são JSON e usam DEFLATE; somente as entradas de diretório
estão armazenadas sem compressão. A razão agregada compactado/descompactado é
aproximadamente 0,1756. O grande outlier de quase 495 MB descompactados torna
arriscado tratar cada thread como uma unidade que sempre cabe na memória.

Uma amostra limitada de um arquivo pequeno de `categories_threads/` confirmou
um objeto com as chaves `category`, `status`, `subcategory`, `threads`,
`total_pages`, `total_threads` e `url`. Os itens de `threads` incluem campos de
identificação, datas, título, tags, contagens, estado e referências de membro.

Uma amostra limitada de um arquivo pequeno de `threads/` confirmou um objeto
com metadados da thread e `messages`. Cada item de `messages` possui as chaves
`creation`, `message`, `official_id`, `user_href` e `user_name`. Os valores não
foram registrados. Portanto, o arquivo preserva metadados ricos, mas também
exige uma política explícita para remover identificadores e referências de
usuário antes de entrar no corpus de treino.

## `conversations.zip`

A enumeração parcial observou uma única árvore:

```text
clear_threads/
  <identificador>.tsv
```

O cabeçalho do ZIP confirma 3.650.595 entradas totais. Antes de a enumeração ser
interrompida, foram observados 1.406.509 membros TSV diretamente sob
`clear_threads/`; nessa parcela não apareceram subdiretórios nem outras
extensões. Não se pode extrapolar essa distribuição para as entradas restantes.
Os nomes sugerem uma exportação já achatada ou limpa de threads, possivelmente
com mais de uma unidade TSV para uma mesma thread, mas essa relação não foi
confirmada sem ler amostras do conteúdo.

Não foi possível confirmar, nesta rodada, o cabeçalho, separador efetivo,
encoding, número de colunas, tamanho descompactado, método de compressão ou
distribuição dos tamanhos dos membros de `conversations.zip`. A leitura do
diretório central por SSHFS ficou sem produzir saída por mais de dois minutos;
a enumeração de nomes foi interrompida depois de 1.406.509 TSVs observados, e a
operação seguinte que agregaria metadados também foi limitada. Assim, não se
deve inferir o schema ou a distribuição completa apenas pela parcela lida.

## Relação observada e decisão vigente

Os dois ZIPs parecem ser representações diferentes do mesmo domínio OuterSpace,
mas a relação não foi provada por comparação de IDs ou conteúdo. O export de
`forum.outerspace.com.br.zip` é rico em categorias, metadados e mensagens, com
unidades JSON de tamanho muito desigual. A exploração indicou que
`conversations.zip` é mais conveniente para ingestão incremental por apresentar
TSVs planos, embora o schema completo ainda precise ser validado sem expor
valores.

A decisão canônica em [docs/datasets/README.md](README.md) seleciona o
`conversations.zip` do OuterSpace. `conversations_min.zip` é somente fixture de
desenvolvimento, e `forum.outerspace.com.br.zip` não é fonte de treino.

Não é necessário reler tudo em cada execução. A recomendação é:

1. gerar uma vez, localmente e fora do Git, um manifesto do diretório central;
2. guardar por membro apenas `member_path`, extensão, tamanho compactado e
   descompactado, CRC, método, data do ZIP e uma classificação de grupo;
3. identificar o arquivo de entrada por tamanho e mtime do mount (e, quando
   houver uma janela de I/O adequada, por hash do ZIP), regenerando o manifesto
   apenas quando esses valores mudarem;
4. fazer uma amostra determinística limitada (por exemplo, alguns membros
   pequenos, medianos e grandes por quantil), registrando apenas cabeçalhos,
   tipos, contagens e erros de parsing;
5. processar em lotes com cursor/resume por membro, mantendo a origem como
   `archive` + `member_path` e escrevendo apenas o derivado necessário.

O manifesto detalhado deve ficar em um diretório local ignorado, não no
repositório junto com datasets. Um relatório agregado como este pode ser
versionado; textos, membros extraídos, caches e pesos não.

## Comandos reproduzíveis e seguros

Os comandos abaixo consultam apenas o diretório central ou membros pequenos;
não devem ser substituídos por `unzip -o` ou extração integral:

```sh
stat -f '%N|%z bytes|%Sm' \
  "$PTBR_DATASET_ROOT/outerspace/forum.outerspace.com.br.zip" \
  "$PTBR_DATASET_ROOT/outerspace/conversations.zip"

# Cabeçalho rápido, sem enumerar milhões de nomes.
zipinfo -h "$PTBR_DATASET_ROOT/outerspace/forum.outerspace.com.br.zip"
zipinfo -h "$PTBR_DATASET_ROOT/outerspace/conversations.zip"

# Para schema, abrir somente membros escolhidos com ZipFile.open(), limitar a
# leitura e descartar os valores; não extrair o arquivo inteiro.
python3 - <<'PY'
import os
from zipfile import ZipFile
from pathlib import Path

archive = (
    Path(os.environ['PTBR_DATASET_ROOT'])
    / 'outerspace/forum.outerspace.com.br.zip'
)
with ZipFile(archive) as z:
    info = z.getinfo('forum.outerspace.com.br/categories.json')
    with z.open(info) as stream:
        sample = stream.read(64 * 1024)
    print(info.file_size, info.compress_size, len(sample))
PY
```

O macOS inspecionado não possui `timeout`/`gtimeout`. Qualquer nova enumeração
potencialmente longa deve usar um wrapper Python com timeout explícito ou ser
executada no host que armazena os ZIPs, onde a leitura do diretório central não
atravessa SSHFS.

No SSHFS atual, `zipinfo` sobre o arquivo de fórum levou aproximadamente 18 s
e a agregação do seu diretório central aproximadamente 20--25 s. Para
`conversations.zip`, a enumeração parcial de nomes consumiu aproximadamente
2--3 min e a agregação completa foi deliberadamente limitada. Esses tempos são mais um
motivo para manter o manifesto local e não repetir a varredura em cada treino.

## Limitações e próxima validação

- Ainda falta confirmar o schema dos TSVs, seu encoding e se cada arquivo é
  uma thread completa ou um fragmento.
- Ainda não foi feita uma comparação de IDs entre os dois ZIPs.
- Não foi calculado o total descompactado de `conversations.zip`.
- Antes de implementar o parser, deve-se ler somente alguns TSVs pequenos e
  médios via `ZipFile.open`, registrar cabeçalho/tipos/contagens agregadas e
  testar se mensagens vazias, HTML, usernames e URLs podem ser removidos de
  forma determinística.
- A quantidade de `conversations.zip` a usar continua pendente. O export JSON
  permanece útil apenas como evidência estrutural; não foi selecionado como
  fonte de treino e contém um outlier de tamanho e campos de usuário que exigem
  sanitização.
