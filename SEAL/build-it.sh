chmod +x configure
sed -i -e 's/\r$//' configure
bash configure
make
