%{
  @article{sela2017unrestricted,
	  title={Unrestricted Facial Geometry Reconstruction Using Image-to-Image Translation},
	  author={Sela, Matan and Richardson, Elad and Kimmel, Ron},
	  journal={arxiv},
	  year={2017}
  }
%}
function [] = geometric_recons(img_path, im_pncc_path, im_depth_path, output)
	%% Process network result (scale and mask)
	img = imread(img_path);
	img = imresize(img, [512 512]);
	im_pncc = imread(im_pncc_path);
	im_depth = imread(im_depth_path);
	im_depth = imresize(im_depth, [512 512]);
	im_pncc = imresize(im_pncc, [512 512]);

	im_depth = im2double(im_depth);
	im_pncc = im2double(im_pncc);
	im_depth = im_depth * 255;
	im_pncc = im_pncc * 255;

	if size(im_depth,3) ~= 3
		im_depth = cat(3, im_depth, im_depth, im_depth);
	end

	
	%{
	class(im_depth)
	class(im_pncc)
    max(max(im_depth))
    max(max(im_pncc))
    %}

    show_figs = false;

	[ Z, pipeline_args ] = raw2depth( img, im_pncc, im_depth ); %% 3, 4
	fprintf('Raw2depth done..... \n')

	%% Apply rigid deformation
	[mesh_result, pipeline_args] = depth2mesh( pipeline_args, show_figs ); %% 5, 6
	fprintf('Depth2mesh done..... \n')

	%% Save depth2mesh (coarse reconstruction)
	vertface2ply(mesh_result.vertex, mesh_result.face, mesh_result.texture, sprintf('../output/%s_depth2mesh.ply', output));

	%% Apply detail extraction

	[ fine_result ] = mesh2fine(pipeline_args); %% 7
	fprintf('Mesh2fine done..... \n')

	%% Save mesh2fine (fine reconstruction)
	vertface2ply(fine_result.vertex, fine_result.face, fine_result.texture, sprintf('../output/%s_mesh2fine.ply', output));
	fprintf('Saving output..... \n')
	
end