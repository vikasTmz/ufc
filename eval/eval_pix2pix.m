function eval_pix2pix(pix2pix_model_result_path)

IMG_DIR = strcat(pix2pix_model_result_path);
imagefiles = dir(sprintf(strcat(IMG_DIR,'*.png')));
nfiles = length(imagefiles);

for ii=1:nfiles
	clearvars -except IMG_DIR ii
	try
		%{
		im_file = imagefiles(ii).name
	    filename_split = strsplit(im_file,'.');
	    filename = char(filename_split(1));
	    filename
	    %}

	    id = char(int2str(ii));
	    

		ground = imread(strcat(IMG_DIR,id,'_real_B.png')); out = imread(strcat(IMG_DIR,id,'_fake_B.png'));
		[ssimval, ssimmap] = ssim(out, ground);

		Gray = rgb2gray(ssimmap);
		RGB1 = cat(3, Gray, Gray, Gray);
		RGB2 = Gray;
		RGB2(end, end, 3) = 0;  % All information in red channel
		GrayIndex = uint8(floor(Gray * 255));
		Map       = jet(255);
		RGB3      = ind2rgb(GrayIndex, Map);

		imwrite(RGB3, strcat(IMG_DIR,id,'_error.png'));

		log_10 = 0;
		MSE = 0;
		wh = numel(ground);
		MSE = immse(ground, out);

		for i = 1:wh
			
			if ground(i) == 0
				ground(i) = 1;
			end

			if out(i) == 0
				out(i) = 1;
			end
			
			log_10 = log_10 + abs(log10(double(ground(i))) - log10(double(out(i))));
		end

		disp(sprintf('%s  %0.4f %0.4f %0.4f', id, ssimval, log_10 / wh, MSE^(1/2)))

    catch ME
        %disp(sprintf('Error'))
    end	
end
