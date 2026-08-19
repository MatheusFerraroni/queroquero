# Adrenaline: exploração inicial dos ZIPs

Data da inspeção: 2026-08-18. O mount externo, representado nos comandos por
`$PTBR_DATASET_ROOT`, foi tratado como somente leitura. Nenhum ZIP foi extraído
e nenhum texto, nome de usuário, URL ou outro valor de registro foi copiado para
este relatório. Os nomes de campos abaixo são apenas metadados de schema.

## Inventário

| Arquivo | Tamanho do ZIP | Entradas no diretório central | Estrutura confirmada |
| --- | ---: | ---: | --- |
| `forum.adrenaline.com.br.zip` | 2.157.145.545 bytes (~2,0 GiB) | 356.799 (356.795 arquivos + 4 diretórios) | 356.795 arquivos `.json`, todos DEFLATE |
| `conversations.zip` | 76.123.303.722 bytes (~71 GiB) | 4.896.419 | Apenas cabeçalho/contagem; enumeração detalhada não concluída no SSHFS |

Os números do primeiro arquivo vieram do diretório central do ZIP e de uma
enumeração de metadados com `zipfile`; não houve leitura do conteúdo completo.
Não há criptografia nem arquivos de tamanho zero no arquivo bruto. Os caminhos
dos arquivos JSON têm profundidade uniforme (dois componentes para 356.793
arquivos e um componente para dois arquivos); os nomes não foram registrados
para evitar transportar identificadores.

### `forum.adrenaline.com.br.zip`

O conjunto contém:

- 356.795 arquivos JSON;
- 12.051.559.230 bytes descomprimidos;
- 2.080.965.959 bytes de payload comprimido;
- método de compressão ZIP 8 (DEFLATE) em todos os arquivos;
- nenhum arquivo vazio ou criptografado.

Foram abertas apenas algumas entradas pequenas, limitadas por tamanho, sem
imprimir seus valores. Os formatos observados foram:

1. Um JSON de topo lista, com objetos contendo `id`, `subs`, `title_href` e
   `title_text`. Os objetos de `subs` observados continham
   `complete`, `description`, `id`, `last_update`, `title_href` e `title_text`.
2. Outro JSON de topo objeto continha `category`, `status`, `subcategory`,
   `threads`, `total_pages`, `total_threads` e `url`. Os itens de `threads`
   observados continham `answers`, `category`, `date_thread`, `href`, `id`,
   `is_fixed`, `last_post`, `member_href`, `member_name`, `subcategory`,
   `tags`, `title` e `visits`.

Isso confirma uma camada de índices/listagens de tópicos e seus metadados. Nos
exemplos limitados, não foi confirmado um corpo de mensagem; alguns campos
aparentam ser contagens ou referências. Também havia entradas individuais com
dezenas de MiB, que não foram carregadas integralmente. Portanto, não se deve
assumir que esse ZIP, sozinho, seja o corpus conversacional pronto para o
treinamento.

### `conversations.zip`

O cabeçalho ZIP confirmou 4.896.419 entradas e o tamanho acima. Tentativas de
carregar o diretório central inteiro via `ZipFile.infolist()` para contar
extensões, somar tamanhos e selecionar amostras permaneceram sem conclusão no
SSHFS por aproximadamente dois minutos cumulativos (cerca de 117 segundos) e
foram interrompidas. Assim, ainda não estão confirmados:

- quantidade por extensão e presença de diretórios;
- tamanhos comprimidos/descomprimidos agregados;
- método de compressão por entrada;
- nomes/padrões dos caminhos internos;
- encoding, delimitador e schema das conversas;
- se é derivado do arquivo bruto, se contém mensagens completas ou se é apenas
  outra etapa intermediária.

Essa limitação é operacional do mount, não uma conclusão sobre o conteúdo do
arquivo.

## Precisamos ler tudo a cada execução?

Não. A leitura completa repetida seria especialmente inadequada para um ZIP de
71 GiB. A primeira exploração deve produzir um manifesto pequeno, persistente
e ignorado pelo Git, contendo:

- tamanho e `mtime` do ZIP, além de um fingerprint do diretório central;
- para cada entrada: índice ordinal, offset, tamanho comprimido e
  descomprimido, CRC, método, extensão e hash do caminho (não o caminho cru);
- amostras limitadas por extensão/tamanho, identificadas por hash do conteúdo e
  com apenas schema, encoding, delimitador e estatísticas agregadas;
- versão do comando/script que gerou o manifesto e timestamp da inspeção.

Nas execuções seguintes, o fingerprint permite reutilizar o manifesto. A
ingestão pode selecionar apenas entradas que atendam ao limite de tamanho,
extensão e política de proveniência, abrindo cada membro diretamente com
`ZipFile.open`. Um cursor por ordinal/offset permite retomar após interrupção;
não é necessário extrair o ZIP nem manter uma cópia descompactada. Se o mount
continuar lento para enumerar `conversations.zip`, o manifesto deve ser gerado
no host que armazena o ZIP e então copiado para o ambiente de execução. Isso
evita transferir repetidamente um diretório central com milhões de entradas
pelo SSHFS. Depois da abertura inicial do diretório central, o processamento
dos membros pode ser feito em lotes persistentes, com checkpoint somente do
progresso e sem recomeçar os lotes concluídos.

## Amostragem segura ainda necessária

Embora a fonte do projeto já esteja selecionada, uma segunda inspeção bounded é
necessária antes de implementar o parser e fixar filtros ou quantidades:

1. Obter somente a lista/metadados do diretório central de
   `conversations.zip`, com timeout curto e progresso persistido.
2. Selecionar no máximo algumas entradas por extensão e por faixas de tamanho.
3. Ler no máximo alguns KiB de cada entrada, detectando encoding, delimitador,
   número de colunas, chaves JSON e distribuição de comprimentos; descartar os
   valores após a medição.
4. Caracterizar a unidade documental (thread, mensagem ou lote), duplicação,
   presença de texto e custo de leitura em relação ao ZIP bruto.

Esta exploração histórica confirmou a estrutura JSON do ZIP bruto, mas encontrou
principalmente metadados de tópicos nas amostras. Ela não confirmou sozinha o
schema completo de `conversations.zip`; essa limitação permanece válida para a
implementação do parser.

## Comandos reproduzíveis e seguros

Os comandos abaixo consultam apenas metadados ou amostras limitadas; não usam
`unzip -o` nem extraem os arquivos:

```sh
ls -lh "$PTBR_DATASET_ROOT/adrenaline/forum.adrenaline.com.br.zip" \
  "$PTBR_DATASET_ROOT/adrenaline/conversations.zip"
zipinfo -h "$PTBR_DATASET_ROOT/adrenaline/forum.adrenaline.com.br.zip"
zipinfo -h "$PTBR_DATASET_ROOT/adrenaline/conversations.zip"
```

Para o arquivo bruto, a enumeração central que produziu as contagens acima usa
`zipfile.ZipFile.infolist()` e soma somente `file_size`, `compress_size`,
`compress_type`, extensão e profundidade do caminho. A leitura de conteúdo foi
feita somente com `ZipFile.open(info).read(min(info.file_size, 32768))` em
entradas pequenas; nenhuma saída de conteúdo foi persistida.

## Limitações e decisão vigente

A decisão canônica em [docs/datasets/README.md](README.md) seleciona o
`conversations.zip` do Adrenaline. `conversations_min.zip` é somente fixture de
desenvolvimento, e `forum.adrenaline.com.br.zip` e `messages.zip` não são fontes
de treino. Ainda é necessário medir schema, texto útil, duplicatas e campos de
proveniência antes de definir um subconjunto configurável, sua quantidade e um
manifesto reutilizável.
