cp /usr/share/ca-certificates/GEExternalRootCA2.1.crt .
docker build . --no-cache -t ailab/geometric-jupyter --build-arg jupyter_password='argon2:$argon2id$v=19$m=10240,t=10,p=8$g+SVn72KJv6CU/1gyO6+oQ$25RBQ69DTiMmKzQ++ZzrqQ' --build-arg proxy=$HTTPS_PROXY

