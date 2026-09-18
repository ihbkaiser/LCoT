pip install gdown

ZIP_NAME="musique_v1.0.zip"
gdown https://drive.google.com/file/d/1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h/view?usp=sharing --output $ZIP_NAME

unzip $ZIP_NAME

rm $ZIP_NAME