# WackyWacky PT-BR

Inventário de 2026-08-18 do arquivo `wacky/pages.tsv` em origem somente
leitura.

| Medida | Valor |
| --- | ---: |
| tamanho | 56.205.366.282 bytes |
| formato | TSV UTF-8 |
| colunas | 19 |
| total de linhas | desconhecido |

Uma amostra histórica limitada de 2.100 linhas em sete offsets encontrou 621
textos. Ela não é usada para estimar o rendimento porque a fonte parece
ordenada por status.

Campos utilizados na seleção: `domain_id`, `status`, `text`, `text_md5` e
`same_as`. URLs, títulos, HTML, timestamps e IDs não entram no texto nem nos
relatórios.

Na fonte, `text` não está em texto puro: contém um frame Zstandard codificado
como hexadecimal. O adapter converte hexadecimal para bytes, descompacta ZSTD,
exige UTF-8 estrito e compara o conteúdo descompactado contra `text_md5` antes
da limpeza e tokenização. Divergências de MD5 observadas na fonte são mantidas
e contabilizadas em `text_md5_mismatches`; payloads inválidos ou maiores que
128 MiB são rejeitados sem registrar conteúdo.

## Seleção

São elegíveis apenas registros que satisfazem todas as regras:

```text
status == "done"
text presente
text_md5 presente
same_as ausente (campo vazio ou marcador literal "NULL")
```

Referências reais em `same_as`, como IDs numéricos, continuam excluídas. O
marcador textual `NULL` representa ausência na fonte e não é tratado como uma
referência.

O perfil `smoke` usa um prefixo limitado para engenharia. O `mvp` faz uma
passagem sequencial completa, calcula o fingerprint durante a leitura e mantém
um reservoir determinístico dentro do budget. O cursor por byte e por registro
permite retomar a passagem; qualquer alteração incompatível da fonte é
rejeitada.

## Gate de boilerplate

No perfil `mvp`, a decisão inicial é `pending`. A execução gera
`boilerplate_report.json`, sem exemplos de texto, e bloqueia a publicação dos
shards finais. O relatório conta parágrafos normalizados com pelo menos 80
caracteres que aparecem em 5 documentos e 3 domínios distintos.

Depois da revisão, altere `filters.boilerplate.decision` em
`configs/datasets/wackywacky.json` para uma opção explícita:

- `keep`: preserva todos os parágrafos;
- `remove_exact`: remove somente os parágrafos que atendem aos limiares
  documentados.

```sh
python -m queroquero.prepare run --dataset wackywacky --profile mvp
```

Deduplicação exata de documentos, limpeza, tokenização, split e packing são
responsabilidade do núcleo comum. A amostra histórica parcial não é usada para
estimar o rendimento global.
