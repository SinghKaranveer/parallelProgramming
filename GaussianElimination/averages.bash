#!/bin/bash
for n in {1..30}; do ./gauss_eliminate.o; done > Output512
cat Output512 | grep 'MT' | sed 's/[^0-9.]//g' | sed 's/.$//' > Output512Final


