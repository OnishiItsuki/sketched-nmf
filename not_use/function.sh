#!/bin/bash
function argarray() {
  begin=$1
  size=$2
  end=$(expr $begin \+ $size)
  shift 2
  i=$begin
  while [ ${i} -lt ${end} ]
  do
    eval echo \$$i
    i=$(expr $i \+ 1)
  done
}
