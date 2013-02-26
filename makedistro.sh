#!/bin/bash

export VER=`grep -o v[0-9]*.[0-9]*.[0-9]* VERSION.TXT`

tar 						\
	--exclude=*/.svn 		\
	--exclude=*/bin 		\
	--exclude=*/testlab 	\
	--exclude=*/graph		\
	--exclude=*/bfs			\
	-czvf back40.$VER.tgz LICENSE.TXT VERSION.TXT back40 test
