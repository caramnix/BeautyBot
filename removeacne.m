

%acne removal- Cara 

% read in images 

img= imread("search_pimple.png");

imagesc(img)
image_R= double(img(:,:,1)); 
image_G= double(img(:,:,2)); 
image_B= double(img(:,:,3));


template = imread("pimple_close_4.png");
imagesc(template)
t_r= double(template(:,:,1)); 
t_g= double(template(:,:,2)); 
t_b= double(template(:,:,3));

%%

for row = 1:size(img,1)
        for column=1:size(img,2)
            M(row,column,:) = [image_R(row,column), image_G(row,column), image_B(row,column)]';
        end 
end 

%%

%define constants for template image
sd_t_r= std(t_r,0,"all"); 
sd_t_g= std(t_g,0,"all"); 
sd_t_b= std(t_b,0,"all");
t_bar_r= mean(t_r, "all"); 
t_bar_g= mean(t_g, "all"); 
t_bar_b= mean(t_b, "all");


%pull out window for each pixel value
%each window is size , with the upperleft-corner as the window origin
c= 0
for row = 1: size(img,1) -size(template,1) +1  %careful indexing to not go out of bounds
    disp(row)
    for column=1:size(img,2) - size(template,2) +1
        patch= M(row:size(template,1)+(row-1),column: size(template,2)+(column-1),:);
        M2(row, column)= calc_NCC(patch, t_r, t_g, t_b, sd_t_r, sd_t_g, sd_t_b, t_bar_r, t_bar_g, t_bar_b); 
    
    end 
 end 

 %%

 %find the MAX value- 
[x2, y2] = nthlargest(M2, 1) 

row= x2;
column= y2;
patch= img(row:size(template,1)+(row-1),column: size(template,2)+(column-1),:);
imagesc(patch)

imagesc(img) 
hold on; 
rowv= x2;%53;
rowchange= (size(template,1)+1); 
colv= y2; %56;
colchange= (size(template,2)+1); 

plot([colv colv+colchange],[rowv rowv],'r','linewidth',.5)
plot([colv colv+colchange],[rowv+rowchange rowv+rowchange],'r','linewidth',.5)
plot([colv colv],[rowv rowv+rowchange],'r','linewidth',.5)
plot([colv+colchange colv+colchange],[rowv rowv+rowchange],'r','linewidth',.5)
hold off;

%%

% now BLUR - sucessive gaussian filtering! 

%%

patch2 = patch; 
patch2= imgaussfilt(patch2,50);
imagesc(patch2)

%%now pop that patch back in 
img_test = img; 
rowv= x2;
colv= y2;
img_test(row:size(template,1)+(row-1),column: size(template,2)+(column-1),:) = patch2; 
imagesc(img_test); 



%% try expanding smoothing window

patch3= img(row-25:size(template,1)+(row-1)+25,column-25: size(template,2)+(column-1)+25,:);

patch3= imgaussfilt(patch3,15);
imagesc(patch3)

%%now pop that patch back in 
img_test = img; 
rowv= x2;
colv= y2;
img_test(row-25:size(template,1)+(row-1)+25,column-25:size(template,2)+(column-1)+25,:) = patch3; 
imagesc(img_test); 



%functions 

function [r,c] = nthlargest(matrix, n)
    for x = 1: n-1 
        [i,j] = find(ismember(matrix, max(matrix(:))));
        matrix(i,j) = -Inf;
    end 
    [r,c]=find(ismember(matrix, max(matrix(:))));
end 

function plot_patch(row,column, image, template)
    patch= image(row:size(template,1)+(row-1),column: size(template,2)+(column-1),:);
    imagesc(patch)
end 


function NCC= calc_NCC(patch, t_r, t_g, t_b, sd_t_r, sd_t_g, sd_t_b, t_bar_r, t_bar_g, t_bar_b)
    p_r= patch(:,:,1); 
    p_g= patch(:,:,2); 
    p_b= patch(:,:,3); 
    
    num_r = (p_r - mean(p_r, "all")) .* (t_r - t_bar_r);
    den_r = std(p_r,0,"all") * sd_t_r; 

    num_g = (p_g - mean(p_g, "all")) .* (t_g - t_bar_g);
    den_g = std(p_g,0,"all") * sd_t_g; 

    num_b = (p_b - mean(p_b, "all")) .* (t_b - t_bar_b);
    den_b = std(p_b,0,"all") * sd_t_b; 

    NCC = sum(sum((num_r/den_r) + (num_g/den_g) + (num_b/den_b)));

end 
