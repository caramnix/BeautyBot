%------------------------RED EYE REMOVER--------------------------------
%Cara  

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


%--------------------------ACNE REMOVER-----------------------------
%Cara 

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


%---------------------PORTRAIT MODE FEATURE----------------------------

% read and display images for background subtraction
mainIm = imread('main.JPG'); 
imagesc(mainIm), axis('image'), title('main image');
pause;

backIm = imread('background.JPG'); 
imagesc(backIm), axis('image'), title('background image');
pause;

% define threshold for background subtraction
T = 10;

% get size of original image to walk through and perform background
% subtraction
[r, c] = size(mainIm);
for i = 1:r
    for j = 1:c
        if (abs(double(mainIm(i, j)) - double(backIm(i, j))) > T)
            B(i,j) = 1;
        else
            B(i,j) = 0;
        end
    end
end

% display subtracted image
imagesc(B), axis('image'), title('subtracted image');
pause;

% erode and display image
B = imerode(B, strel('disk',1));
imagesc(B), title('eroded image');
pause;

% dilate and display eroded image
B = bwmorph(B, 'dilate'); 
imagesc(B), axis('image'), title('dilated background subtracted image');
pause;

% keep only connected components of dilated image
B = bwareaopen(B, 500);
imagesc(B), title('connected components image'); 
pause;

% create blurred background using gaussian smoothing
sigma=15;
G = fspecial('gaussian', 2*ceil(3*sigma)+1, sigma);
backgroundBlur = imfilter(backIm, G, 'replicate');
imagesc(backgroundBlur);
pause;

% get blurred background for final portait
portrait = backgroundBlur;

% loop through each pixel of the background subtracted image and if the
% pixel is 1, then it is part of the main content so put it on top of the
% blurred background
for i = 1:r
    for j = 1:c
        if (B(i,j)==1)
            portrait(i,j) = mainIm(i,j);
        end
    end
end

% display final portrait
imagesc(portrait), title('portrait');
pause;

%-----------------------LIP COLOR CHANGER-------------------------------

% read in image to search from, get size of dimensions, and display
face = imread("woman_face_1.jpg");
[sr, sc, sz] = size(face);
imagesc(face), title('model to change lip color on');
pause;

% read in image template to find, get size of dimensions
lips = imread("woman_lips_1.jpg");
[tr, tc, tz] = size(lips);


% initialize variable to hold SSD value
SSD = 0;

% initialize matrices to hold SSD values and their corresponding
% pixel origin locations
ssdArray = [];
ssdLocations = [];

% loop through search image 
for rr=1:sr-(tr-1)
    for cc=1:sc-(tc-1)

        % calculate index for end row/col of patch
        endRow = rr+(tr-1);
        endCol = cc+(tc-1);


        % get (next) patch of image
        candPatch = face(rr:endRow, cc:endCol, :);
       
        % calculate dimensions of candidate patch obtained
        % will end up being template dimensions, but could use this if
        % jumping windows and edge windows may end up diff sizes
        r = endRow - rr + 1;
        c = endCol - cc + 1;

        % set up / clear matrix to store <R, G, B> values for each
        % pixel in candidate patch
        featureRows = r*c;
        res = zeros(featureRows,3);

        % initialize / reset index to use for feature vector matrix
        resIdx = 1;

        % loop through all rows and columns of candidate patch
        for i=1:r
            for j=1:c
        
                % perform SD calculation for each color channel
                res(resIdx,1) = (double(candPatch(i,j,1)) - double(lips(i,j,1)))^2;
                res(resIdx,2) = (double(candPatch(i,j,2)) - double(lips(i,j,2)))^2;
                res(resIdx,3) = (double(candPatch(i,j,3)) - double(lips(i,j,3)))^2;

                resIdx = resIdx + 1;

            end
               
        end

        
        % sum SD among each color channel to get SSD
        SSD = sum(sum(res));


        % store SSD and pixel origin of patch it came from
        ssdArray(end+1) = SSD;
        ssdLocations(end+1,:) = [rr cc];

        % reset SSD variable
        SSD = 0;

    end
end

% min SSD value = best match
ssdVal = min(ssdArray);

% find row/col location of best SSD value in SSD array
[row, col] = find(ismember(ssdArray, ssdVal));

% col of best SSD in SSD array corresponds to row containing
% patch origin location in location vector, so obtain location of best SSD 
% patch using col
searchImgMatchLocation = ssdLocations(col, :);
matchRow = searchImgMatchLocation(1);
matchCol = searchImgMatchLocation(2);

% obtain and display matched image portion
matched = face(matchRow:matchRow+(tr-1), matchCol:matchCol+(tc-1), :);

% compute edge of matched region using canny detection
myEdge = edge(rgb2gray(matched), 'canny');
imagesc(myEdge), title('edge of matched lips');
pause;

% find connected components to get more accurate area
myEdge = bwareaopen(myEdge, 8);
imagesc(myEdge), axis('image'), title('connected components edge');
pause;

% loop through the isolated matched region (same size as template) to 
% color lips new color
for i=1:tr-1
    for j=1:tc

        % keep track of if we are within the lip area
        inLips = false;

        % if the corresponding point in edge boundary is 1, this is an edge
        % of the lip, so color it the new color
        if (myEdge(i,j) == 1)

            %to change the color realistically, vary the channels of the
            %old color (so it isn't a flat value for the whole lip area)
            % -making the lip color more purple in this case
            matched(i,j,1) = matched(i,j,1);
            matched(i,j,2) = matched(i,j,2);
            matched(i,j,3) = matched(i,j,3) + 50;

            % if the next edge px is 0, we are either in the lips or there 
            % is no lip left (corner or thin edge of lip)
            if (myEdge(i+1,j)==0)

                % find if there is more lip left by seeing if there is
                % another edge left in the columns of the current row
                edgesLeft = find(myEdge(i+1:tr-1, j) == 1);
                
                % if there is an edge left, we are in the lips so 
                % initialize variable to keep track of row index
                if (sum(edgesLeft)>0 )
                    inLips = true;
                    k=i+1;
                end

            end
        end

        while (inLips)
            
            % if current and next px are zero, we are fully in lip area
            if(myEdge(k,j) == 0 && myEdge(k+1, j) == 0)
                matched(k,j,1) = matched(k,j,1);
                matched(k,j,2) = matched(k,j,2);
                matched(k,j,3) = matched(k,j,3) + 50;

            % if current px is zero and next px is one, we are about to be
            % out of lip area
            elseif(myEdge(k,j) == 0 && myEdge(k+1, j) == 1)
                matched(k,j,1) = matched(k,j,1);
                matched(k,j,2) = matched(k,j,2);
                matched(k,j,3) = matched(k,j,3) + 50;
                inLips = false;
            end

            % if we are still in the loop incremement row count
            k = k + 1;
        end
    end
end

imagesc(matched), title('colored lips');
pause;

% put the newly colored lips back on the model and display
face(matchRow:matchRow+(tr-1), matchCol:matchCol+(tc-1), :) = matched;
imagesc(face), title('colored lips on model');

%--------------------functions-----------------------------
function results= g_blurr(image, r,c )
    blurr_sum= 4*image(r,c) + 2* image(r-1,c) + 2* image(r+1,c) + 2*image(r,c-1) + 2*image(r,c+1) + image(r-1,c-1)  + image(r-1,c+1) +  image(r+1,c+1) + image(r+1,c-1)   ;         
    results= (blurr_sum/16);
    
end 

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