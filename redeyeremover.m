

%removing red eye - Cara  

%read in image
img= imread("red_eye_ex1.jpg");

imagesc(img); 

%set RGB values into separate matrices
image_R= double(img(:,:,1)); 
image_G= double(img(:,:,2)); 
image_B= double(img(:,:,3));

%looking at just the R section, can we isolate the red eyes w/ edge
%detection? 

%% use Canny Edge Detection on R,G.B matrices

colormap(gray); 

edD= edge(image_R,'canny', .66); %.66; 
imagesc(edD);


edD1= edge(image_G,'canny', .55); 
imagesc(edD1);


edD2= edge(image_B,'canny', .55); 
imagesc(edD2);

%%

%so we've ID'ed the eyes using canny edge detection- but also the eyebrows & some hair

%combine them into one image 
combine = edD + edD1 +edD2; 

%now only save pixels which were Id'd as edges in 2 or more of the RGB
for row = 1:size(combine,1)
    for col = 1:size(combine,2)
        if combine(row, col) >= 2 
            c(row,col) =1 ; 
        else 
            c(row,col) = 0;
        end 

    end
end 

% 1 means it's an edge in one of the images 
% 2 means it's an edge in two of the images 
% 3 means it's an edge in three of the images 


%%
imagesc(c)

%use close technique 

closeBW = imclose(c,strel('disk',20));
figure, imshow(closeBW)

%%

%boom, now we know where to look for the eyes-- so now we must go through
%and find the red spots 

imagesc(img)
hold on 
openBW= bwareaopen(closeBW, 600);
imagesc(openBW)

%%

%save these coordinates 

i=1;
for row = 1:size(openBW,1)
    for col = 1:size(openBW,2)
        if openBW(row, col) >0 
            x_coords(i) =row ; 
            y_coords(i) =col; 
            i = i+1; 
        end 
    end
end 

%%
clf;
imagesc(img)
hold on
scatter(y_coords, x_coords); 

%%

%okay so now that we have the coordinates, we are going to want to find the
%actual spot that is red


closeBW2 = imdilate(openBW,strel('disk',20));
figure, imshow(closeBW2)

i=1;
for row = 1:size(closeBW2,1)
    for col = 1:size(closeBW2,2)
        if closeBW2(row, col) >0 
            x_coords2(i) =row ; 
            y_coords2(i) =col; 
            i = i+1; 
        end 
    end
end 


clf;
imagesc(img)
hold on
scatter(y_coords2, x_coords2); 



%%
j=1; 
for row = 1:size(y_coords2,2)
   test = image_R(x_coords2(row), y_coords2(row)); 
   if test > 140 
       if  image_G(x_coords2(row), y_coords2(row)) < 80
           if  image_B(x_coords2(row), y_coords2(row)) <80
                R_vals2(j) = test; 
                x_coords_short(j)= x_coords2(row);
                y_coords_short(j)= y_coords2(row);
                j= j+1; 
           end 
       end 

   end  
end 

%%

clf;
imagesc(img)
hold on
scatter(y_coords_short, x_coords_short, "."); 

%WOOOO identified what we need y_coords_short,x_coords_short 

%%

%Now, let's fix it 

%img= imread("red_eye_ex1.jpg");
%image_R= double(img(:,:,1)); 
image_R_2= double(img(:,:,1));

for row = 1:size(y_coords_short,2)
    image_R_2(x_coords_short(row), y_coords_short(row))=  image_R(x_coords_short(row), y_coords_short(row))-120; 
end 



%%
image_new(:,:,1)= uint8(image_R_2); 
image_new(:,:,2)= uint8(image_G); 
image_new(:,:,3)= uint8(image_B); 

image_new= uint8(image_new); 

imagesc(image_new)


%% OPT- BLUR 

image_R_3= image_R_2; 
image_G_3 = image_G; 
image_B_3 = image_B; 

for row = 1:size(y_coords_short,2) %for each coord, pull surrounding pixels & blurr
    r= x_coords_short(row); 
    c = y_coords_short(row);
    image_R_3(r,c)= g_blurr(image_R_2, r,c); 
    image_G_3(r,c)= g_blurr(image_G, r,c); 
    image_B_3(r,c)= g_blurr(image_B, r,c); 
    image_R_3(r, c)=  image_R_3(r,c)-120; 
    % blurr_sum= 4*image_R_2(r,c) + 2* image_R_2(r-1,c) + 2* image_R_2(r+1,c) + 2*image_R_2(r,c-1) + 2*image_R_2(r,c+1) + image_R_2(r-1,c-1)  + image_R_2(r-1,c+1) +  image_R_2(r+1,c+1) + image_R_2(r+1,c-1)   ;         
    %image_R_3(r,c)= (blurr_sum/16);
end 

image_blurred(:,:,1)= uint8(image_R_3); 
image_blurred(:,:,2)= uint8(image_G_3); 
image_blurred(:,:,3)= uint8(image_B_3); 

image_blurred = uint8(image_blurred); 

imagesc(image_blurred)


function results= g_blurr(image, r,c )
    blurr_sum= 4*image(r,c) + 2* image(r-1,c) + 2* image(r+1,c) + 2*image(r,c-1) + 2*image(r,c+1) + image(r-1,c-1)  + image(r-1,c+1) +  image(r+1,c+1) + image(r+1,c-1)   ;         
    results= (blurr_sum/16);
    
end 















