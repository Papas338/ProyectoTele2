%% Captura reconocimiento de caras a traves de la camara web.
%Limpiar el Workspace
clear all
close all
clc

%% Crear el objeto detector de cara.
% Este es obtenido gracias a la libreria vision toolbox de MATLAB
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MinSize',...
    [150,150]);

%% Aqu� n la cantidad de veces que se ejecuta el ciclo, se puede cambiar
% el umbral (n) en funci�n de la cantidad de datos que se necesite
n = 400;

%% Cambie str a s01, s02, s03, .... para guardar hasta cu�ntos individuos
% desee guardar en las carpetas respectivas con la funci�n imwrite en 
% la l�nea 138.
str = 'muestra';

%% Se crear el objeto rastreador de puntos.
% Estos seran los que apareceran mientras este activa la camara web
% indicando los sectores claves para la detecci�n de la cara.
% pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

%% Se crear el objeto de la c�mara web.
cam = webcam();

%% Se captura un frame (cuadro/imagen) de video de la camara web para
% obtener el tama�o que debe tener el video.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

%% Se crear el objeto de reproducci�n de video.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2),...
    frameSize(1)]+30]);
runLoop = true;
numPts = 0;
frameCount = 0;
i=1;

%% Se inicia el ciclo para la reproducci�n de video por la camara web y
% para obtener las imegenes de caras de los individuos
while runLoop && frameCount < n

    % Se obtiene el siguiente frame (cuadro/imagen) de la camara web y se
    % guarda para ser despues reproducido en video.
    videoFrame = snapshot(cam);
    
    % Se debe convertir el frame (cuadro/imagen) de la camara web a escala
    % de grises para poder ser analizado.
    videoFrameGray = rgb2gray(videoFrame);
    frameCount = frameCount + 1;
    % Se crea la carpeta en la que estaran guardadas las imagenes
    % con la cara detectada.
    mkdir('photos',str);

    % Se guardan las imagenes.
    imwrite(videoFrame,fullfile('photos',str,[int2str(i),'.jpg']));
    
    i = i + 1;

    % Se visualiza el cuadro de video guardado utilizando el objeto
    % reproducci�n de video.
    step(videoPlayer, videoFrame);

    % Se comprueba si la ventana del reproductor de video se ha cerrado.
    runLoop = isOpen(videoPlayer);
end

%% Se limpian las variables de video.
clear cam;
release(videoPlayer);
release(faceDetector);