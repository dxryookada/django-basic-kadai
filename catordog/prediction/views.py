from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            # model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'model.h5')
            model = load_model(model_path)
            result = model.predict(img_array)
            if result[0][0] > result[0][1]:
                prediction = '猫'
            else:
                prediction = '犬'        
            # 4章で、画像ファイル（img_file）の前処理を追加
            # 4章で、判定結果のロジックを追加
            # 暫定で、ダミーの判定結果としてpredictionにランダムで「猫」か「犬」を格納
            # prediction = random.choice(["猫", "犬"])
            return render(request, 'home.html', {'form': form, 'prediction': prediction})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})
