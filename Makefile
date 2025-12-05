get_data:
	bash ./src/rita.sh

unzip_data:
	for z in .src/2024_*.zip; do 
    	unzip -j "$z" '*.csv'