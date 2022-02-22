#!/usr/bin/env bash
rm ./tutorials.rst
cp ./tutorials_template ./tutorials.rst
echo " " >> ./tutorials.rst
# jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' --to rst ../notebooks/overview.ipynb
for notebook in `ls ../notebooks/tutorials/*.ipynb | sort -g` ; do
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' --to rst $notebook
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' --to rst $notebook
    jupyter nbconvert --TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' --to rst $notebook
    filename1=${notebook##*/}
    filename=${filename1%??????}
    echo "    tutorials/$filename" >> ./tutorials.rst
done
rm -r ./tutorials
mkdir ./tutorials
mv ../notebooks/tutorials/*.rst ./tutorials
cp ./static_tutorials/*.* ./tutorials
for fils in ../notebooks/tutorials/*_files ; do
    mv $fils ./tutorials
done
# mv ../notebooks/overview.rst .
mkdir ./tutorials/notebooks
for notebook in ../notebooks/tutorials/*.ipynb ; do
    cp $notebook ./tutorials/notebooks
done

for rstfile in ./tutorials/*.rst; do
    filename1=${rstfile##*/}
    filename=${filename1%????}
    echo "" >> $rstfile
    # echo ".. only:: builder_html">> $rstfile
    echo "" >> $rstfile
    echo "Download :download:\`Notebook<notebooks/$filename.ipynb>\`." >> $rstfile
    echo "" >> $rstfile
    echo "" >> $rstfile
done
echo "------------------------------------------------"
make clean
rm -r api
mkdir ./tutorials
for ext in png jpg gif pdf; do
	echo ../notebooks/tutorials/*.$ext ./tutorials/
	cp ../notebooks/tutorials/*.$ext ./tutorials/
	for fils in ./tutorials/*_files ; do
		echo ../notebooks/tutorials/*.$ext ./tutorials/$fils/
		cp ../notebooks/tutorials/*.$ext ./tutorials/$fils/
	done
done
