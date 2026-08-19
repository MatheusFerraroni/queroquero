# WackyWacky PT-BR

Inspeção de 2026-08-18 do arquivo `wacky/pages.tsv` em mount somente leitura.

| Medida | Valor |
| --- | ---: |
| tamanho | 56.205.366.282 bytes |
| formato | TSV UTF-8 |
| colunas | 19 |
| total de linhas | desconhecido |

Campos relevantes: `id`, `domain_id`, `url_md5`, `status`, `text`,
`text_md5`, `same_as` e timestamps. URLs e títulos não entram no texto.

Uma amostra limitada de 2.100 linhas em sete offsets encontrou 621 textos. Dos
registros `done`, 568 tinham texto; a amostra não é representativa porque o
arquivo parece ordenado por status.

## Seleção inicial

```text
status == "done"
text presente
text_md5 presente
```

Registros bloqueados, falhos, pendentes, sem texto ou apenas `same_as` ficam
fora por padrão.

## Antes da preparação

1. fazer uma única passagem sequencial e retomável por byte;
2. contar status, textos, tokens e duplicação por `text_md5`;
3. medir idioma, boilerplate, qualidade e distribuição por domínio;
4. definir budget e gerar shards com proveniência;
5. invalidar derivados se fonte ou configuração mudar.

Não usar a amostra parcial para estimar o rendimento global.
