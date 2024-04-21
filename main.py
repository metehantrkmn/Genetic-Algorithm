
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import structural_similarity as ssim
from skimage import io
from PIL import Image, ImageDraw
import os


"""
Bu fonksiyon verilmiş olan bir kromozom ile girdi olarak verilen resmi ayni karede çizer
Böylece çizgiler ile resim arasindaki benzerlikler daha rahat gorulebilir
Egitim suresinin kisaltilabilmesi icin fitness fonksiyonu kucuk boyutlu resmi input olarak alir
Goruntu kalitesinin daha iyi olabilmesi icin mevcut sonuc alinip daha kaliteli resim ile birlikte cizilir

fonksiyon almis oldugu kromozom icerisindeki noktalari sirasiyle cizgi seklinde birlestirecek sekilde calisir
bunu yapabilmek icin resmin boyutlarina gore bir daire cizer ve devaminda trigonometrik fonksiyonlardan yararlanarak koordinatlari bulur
"""
def polar_grafik_ciz(dizi,input_image_path,iterasyon=0):
    
    aggregate = "k_"
    
    k_image_path = aggregate + input_image_path

    #reads the image
    input_image = Image.open(k_image_path)

    combined_image = input_image.copy()

    #Creates an object for drawing given image
    draw = ImageDraw.Draw(combined_image)

    #get the sizes of image
    width, height = input_image.size

    #coorinates of the center
    center_x = width / 2
    center_y = height / 2

    radius = min(width, height) / 2

    #draws an ellipse on input_image
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline='red')

    #takes every two sequential value in the array and draw a line calculating the coordinates using trigo functions
    for i in range(len(dizi) - 1):
        start_deg = dizi[i]
        end_deg = dizi[(i + 1) % 360]  # Wrap around to the first point

        start_point = (center_x + radius * np.cos(np.radians(start_deg)),
                       center_y + radius * np.sin(np.radians(start_deg)))

        end_point = (center_x + radius * np.cos(np.radians(end_deg)),
                     center_y + radius * np.sin(np.radians(end_deg)))

        draw.line([start_point, end_point], fill="blue", width=2)

    parts = input_image_path.split(".")
    folder_name = parts[0]  

    combined_image.save(f"{folder_name}/output{iterasyon}.png")


def plot_polar_graph(ax, angles, lines):
    for j in range(lines):
        start_angle = np.deg2rad(angles[j])
        end_angle = np.deg2rad(angles[(j + 1) % lines])
        x = [start_angle, end_angle]
        y = [1, 1]
        ax.plot(x, y, color='black', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)


"""
Bu fonksiyon kromozomlarin uygunluklarini puanlayan fonksiyondur
Kromozom degerlerine gore solution.png adli bir dosyaya trigo fonksiyonlarindan yararlanarak cizim yapar
Devaminda hem input dosyasini hem de solution.png dosyasini opencv yardimiyla okuyup numpy araciligi ile array haline getirilir
Elde edilen arraylar korelasyon fonksiyonuna tabii tutulur ve elde edilen deger geri dondurulur
"""
def calculate_fitness(kromozom, num_points, input_img):
    resim2 = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)
    
    width, height = resim2.shape

    #daireyi ve noktalari ayarlar
    circle_radius = np.sqrt(width ** 2 + height ** 2) / 2
    center_x = width / 2
    center_y = height / 2
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points_on_circle = [(center_x + circle_radius * np.cos(angle), center_y + circle_radius * np.sin(angle)) for angle in angles]

    solution_img = Image.new('L', (width, height), color=255)

    #noktalari birlestir
    for i in range(len(kromozom) - 1):
        start_point = points_on_circle[kromozom[i] - 1]
        end_point = points_on_circle[kromozom[i + 1] - 1]
        draw = ImageDraw.Draw(solution_img)
        draw.line([start_point, end_point], fill=0, width=1)

    solution_img = solution_img.resize((width, height))
    solution_img.save("try.png")

    solution_array = np.array(solution_img).flatten()

    reference_array = np.array(resim2).flatten()

    korelasyon_katsayisi = np.corrcoef(solution_array, reference_array)[0, 1]
    return korelasyon_katsayisi

#jenerasyonu input olarak alir ve butun kromozomlari sirasiyla fitness fonksiyonuna tabi tutup uygunluk degerlerini array icerisinde dondurur
def fonkhesap(all_generation, points_of_generation, num_of_pop, input_image_path):

    for i in range(num_of_pop):
        benzerlik_orani = calculate_fitness(all_generation[i], 360, input_image_path)
        points_of_generation[i] = benzerlik_orani

#Random degerlerle indis belirler ve ona gore iki kromozomun farkli parcalarirni birlestirerek yeni kromozomlar uretir
def crossover(dizi1, dizi2):
    uzunluk = len(dizi1)
    yeni_dizi = np.zeros(uzunluk)

    # Çaprazlama noktasını rastgele seç
    caprazlama_noktasi = np.random.randint(1, uzunluk)

    # Eşit olmadığı sürece rastgele seç
    while dizi1[caprazlama_noktasi - 2] == dizi2[caprazlama_noktasi - 1]:
        caprazlama_noktasi = np.random.randint(1, uzunluk)

    # Yeni diziyi oluşturma
    yeni_dizi[:caprazlama_noktasi] = dizi1[:caprazlama_noktasi]
    yeni_dizi[caprazlama_noktasi:] = dizi2[caprazlama_noktasi:]

    return yeni_dizi


"""
Bu fonksiyon siralama secimini kullanarak uygunluk degeri yuksek olanlar icin daha yuksek ihtimalle secim yapan fonksiyondur
Fonksiyonda siralama seciminde oldugu gibi uygunluk degerlerini siralar ve ihtimalleri indis degerleri /toplam degerler olarak belirler
Devaminda random fonksiyonundan uygun bir sekilde yararlanabilmek icin bu ihtimallerin kumulatif toplamlarindan yeni bir dizi uretir 
Boylece random fonksiyonunun uretecegi degerler kumulatif toplam array'i uzerinde uygun raliga uygun ihtimallerde dusecektir
"""
def rastgele_sec(dizi,num_of_pop):

    sorted_array = np.sort(dizi)

    new_fitness_values = np.arange(1,len(sorted_array) + 1)

    probabilities = new_fitness_values / np.sum(new_fitness_values)

    cumulative_sum_probabilities = np.cumsum(probabilities)

    #a number between 0-1
    random_probability = random.random()

    index = 0
    #find the slot where random number hits
    for i in range(len(cumulative_sum_probabilities)):
        if(random_probability >= cumulative_sum_probabilities[i]):
            index = i + 1

    value = sorted_array[index]
    returned_index = np.where(dizi == value)[0][0]

    return returned_index


#verilen mutasyon oranina göre random bir sayi araciligiyla mutasyon gerceklestirir
def degisiklik_yap(dizi,mutation_rate):

    for k in range(num_of_pop):
        for l in range(num_of_cromosome):
            if np.random.rand() < mutation_rate:
                # Değiştirilecek yeni değeri 0 ile 360 arasında rastgele seçelim
                yeni_deger = random.randint(0, 360)

                #uretilen noktanin ayni nokta olmadigindan emin olur
                if ((l != 0 and yeni_deger != dizi[k][l - 1]) or l == 0):
                    dizi[k][l] = yeni_deger
                    
    return dizi


"""
program hangi fotograf icin calistirilcaksa onun ismi yada yolu input_image_path degiskenine girilir

yine uygulanacak olan parametreler icin asagidaki degerler girilir

program her calistirildiginda resimin isimi kullanılarak bir klasor olusturulur ve output'lari oraya yerlestirir
"""

input_image_path = "yildiz.png"

num_of_pop = 800
num_of_cromosome = 60
num_of = 0
mutation_rate = 0.015
total_iteration_number = 1000
all_generation = np.random.randint(0, 359, size=(num_of_pop, num_of_cromosome), dtype=int)

#kalsor ismi icin split yap
parts = input_image_path.split(".")
folder_name = parts[0]  

#yeni klasor olustur
folder_path = folder_name
os.makedirs(folder_path)

mean_s = []
max_values = []

#ilk jenerasyon icin random kromozomlar uretilir
for i in range(num_of_pop):
    for j in range(num_of_cromosome):
        deger = random.randint(0, 360)
        if (j != 0):
            while (deger == all_generation[i][j - 1]):
                deger = random.randint(0, 360)
        all_generation[i][j] = deger

points_of_generation = np.zeros(num_of_pop)
iterasyon = 1

#genetik algoritma belirlenen itersayon sayisinca calistirilir
while (num_of < total_iteration_number):
    
    #jenerasyondaki kromozmlari tek tek fitness fonksiyonuna veren fonksiyon
    fonkhesap(all_generation, points_of_generation, num_of_pop, input_image_path)
    num_of = num_of + 1
    tmp_all_generation = all_generation

    #elde edilen uygunluk degerleri uzerinden crossover icin rastgele secim yapan fonksiyon
    #uygunluk degeri yuksekse secilme ihtimali daha yuksek
    for i in range(num_of_pop):
        ilk = rastgele_sec(points_of_generation,num_of_pop)
        iki = rastgele_sec(points_of_generation,num_of_pop)
        yedekyedek = crossover(all_generation[ilk], all_generation[iki])
        tmp_all_generation[i] = yedekyedek

    all_generation = tmp_all_generation
    #belirli oranda mutasyonu uygulayan fonksiyon
    all_generation = degisiklik_yap(all_generation,mutation_rate)
    
    
    #degerlerin bastirilmasi
    max_index = np.argmax(points_of_generation)  # argmax en buyuk elemanın indexini dondurur
    print(iterasyon)
    print("max index = " + str(max_index))
    print("deger " + str(points_of_generation[max_index]))
    mean = np.mean(points_of_generation)
    
    mean_s.append(mean)#baslangıc burada
    max_values.append(points_of_generation[max_index])
    
    print("mean" + str(mean))
    iterasyon = iterasyon + 1
    # print(all_generation[max_index])
    if (num_of % 10 == 0):
        polar_grafik_ciz(all_generation[max_index], input_image_path, num_of)
        
fig, axs = plt.subplots(2, figsize=(8, 6))  # 2 satır, 1 sütunlu bir alt grafik düzeni

# Mean grafiğini oluştur
axs[0].plot(mean_s, color='blue', marker='o', linestyle='-', label='Mean Values')
axs[0].set_xlabel('Index')  # X ekseni etiketi
axs[0].set_ylabel('Mean Values')  # Y ekseni etiketi
axs[0].set_title('Mean Values by Index')  # Grafiğin başlığı
axs[0].legend()  # Gösterilen verilerin açıklamalarını ekle
axs[0].grid(True)  # Izgarayı göster

# Max values grafiğini oluştur
axs[1].plot(max_values, color='red', marker='s', linestyle='--', label='Max Values')
axs[1].set_xlabel('Index')  # X ekseni etiketi
axs[1].set_ylabel('Max Values')  # Y ekseni etiketi
axs[1].set_title('Max Values by Index')  # Grafiğin başlığı
axs[1].legend()  # Gösterilen verilerin açıklamalarını ekle
axs[1].grid(True)  # Izgarayı göster

plt.tight_layout()  # Grafikler arasındaki boşluğu ayarla
plt.show()  # Grafikleri göster