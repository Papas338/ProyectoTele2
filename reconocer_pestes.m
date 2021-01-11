%% Reconocimiento de plagas
%Limpiar el Workspace
clear all
close all
clc

%% n es el número de plagas a detectar
n = 2;

%% Se carga el database
database = imageDatastore('Img','IncludeSubfolders',true,...
    'LabelSource','foldernames');
% empieza con el nombre de la carpeta, luego si busca en subcarpetas (true
% o false, los label estan dado por el nombre de la carpeta

%% Montaje en pantalla de la primer imagen de cada subtipo
figure;
subplot(1,2,1);montage(database.Files(1));
title('Imágen de una cucaracha');
subplot(1,2,2);montage(database.Files(1+759));
title('Imágen de una rata');

 %% Mostrar imagen de consulta y base de datos lado a lado
personToQuery = 15;
figure;
subplot(2,2,1);montage(database.Files(personToQuery));
title('Imágen de una cucaracha');
subplot(2,2,3);montage(database.Files(personToQuery+759));
title('Imágen de una rata');
subplot(2,2,[2 4]);montage(database);
title('Todo el database');

%% Se parte la base de datos en sección de entrenamiento y test en una
% relación de 80 a 20
[Training ,Test] = splitEachLabel(database,0.8,'randomized');

%% Se copia las capas de Alexnet y se modifican las capas 23 y 25 
% La capa 23 por una "capa totalmente conectada" que recibe la cantidad de
% pestes que tiene el database
% La capa 25 por una "capa de clasificación", esta a su vez sera la ultima
% capa o, capa de salida"
fc = fullyConnectedLayer(n); % crea capa de conexión (n = numero de 
% divisiones del database)
net = alexnet;
ly = net.Layers; % copia de las capas de la red
ly(23) = fc; %se cambia la capa 23 por la nueva capa de conexion
%por que la capas de alexnet por defecto es el usado para imagenet
cl = classificationLayer; %capa de clasificacion
ly(25) = cl;

%% Se configuran las opciones para entrenar la red como:
% La tasa de aprendizaje, la cantidad de epocas de entrenamiento y minimo
% tamaño de lote (cantidad de ejemplos de capacitación utilizados en una
% iteración, mayor que 1 pero menor que la cantidad de datos)
% Una configuración importante a tener en cuenta es el primer parametro del
% trainingOptions, estos pueden ser 'sgdm' | 'rmsprop' | 'adam', eh indican
% que tipo optimizador por descenso del gradiente se usara.

learning_rate = 0.00001; %movimiento dentro de la gradiente desde el punto
opts = trainingOptions("sgdm","InitialLearnRate",learning_rate,...
    'MaxEpochs',20,'MiniBatchSize',64,'Plots','training-progress');
%remsprop metodo de entrenamiento (puede cambiarse)
%Maximas epocas = entrena por generaciones, osea iteraciones de
%entrenamiento
%minibatch size buscar, plots buscar
[newnet,info] = trainNetwork(Training, ly, opts);
%newnet = nueva red

%% Se clasifican los datos de prueba con la red entrenada
[predict,scores] = classify(newnet,Test);

%% Se mide la precisión de la red entrenada con los datos de prueba
% Se toma las etiquetas del conjunto de pruebas y se guardan en names
names = Test.Labels;

% Se guardan en pred las coincidencias entre las etiquetas del conjunto de
% prueba y las que arrojo la red neuronal.
pred = (predict==names);

% Se hace el calculo del porcentaje de precisión sumando la cantidad de
% aciertos y dividiendolo entre la cantidad de datos (Imagenes)
s = size(pred);
acc = sum(pred)/s(1);

% Se imprime en pantalla el porcentaje de precisión
fprintf('La precisión del conjunto de prueba es %f %% \n',acc*100);

%% Probar la red neuronal con una imagen

camara = imageDatastore('photos','IncludeSubfolders',true,...
    'LabelSource','foldernames');
% El valor de face sera 1 si detecta una cara, en caso contrario sera 0
%% Se muestra en pantalla la cara detectada y el histograma de
% gradientes orientados (HOG)
figure;
subplot(1,2,1);imshow(img);
title('Cara de entrada');
subplot(1,2,2);imshow(img);
hold on;plot(visualization);title('Descriptor de características HOG');

%% Se cambia el tamaño de las imagenes al tamaño de entrada de la red.
camara.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

%% Se introduce la imagen a la red neuronal para ser analizada.
predictFace = classify(newnet,camara);

%% Mensaje que saldra al detectar un rostro
nameOfEmotion01 = 'Cucaracha';
nameOfEmotion02 = 'Rata';

%% Se comprueba que tipo de emoción es la encontrada y saca un mensaje
% en consola del tipo de cara detectada.
if predictFace == nameOfEmotion01
    fprintf('La peste detectada es una %s',nameOfEmotion01);
elseif  predictFace == nameOfEmotion02
    fprintf('La peste detectada es una %s',nameOfEmotion02);
end

%%
% podemos usar [predict,score] = classify(newnet,img) aquí puntaje dice el
% porcentaje que cuán confianza es