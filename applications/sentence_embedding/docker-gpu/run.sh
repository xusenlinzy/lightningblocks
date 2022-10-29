docker build -t bertvec-gpu:v1 .

docker run -d -it -p 4001:4000 --name="bertvec-gpu" --gpus=all bertvec-gpu:v1
