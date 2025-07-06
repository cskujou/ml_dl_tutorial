cd dataset
# git clone https://github.com/pratikkayal/PlantDoc-Dataset.git
mkdir -p "PlantDoc-Dataset/test/Tomato two spotted spider mites leaf"
cd ..
mkdir model
for model_name in "timm/resnet50.a1_in1k" "timm/swinv2_base_window8_256.ms_in1k" "timm/convnextv2_base.fcmae_ft_in22k_in1k_384"
do  
    echo Downloading $model_name
    huggingface-cli download $model_name --local-dir model/$model_name
done