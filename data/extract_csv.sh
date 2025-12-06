for zipfile in *.zip; do
  unzip -j "$zipfile" '*.csv'
done