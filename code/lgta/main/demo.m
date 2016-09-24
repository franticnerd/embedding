function e = demo(input_dir, output_dir, n_region, n_topic)

location_file = '../data/car/car_image_location.txt'
tag_file = '../data/car/car_image_tags.txt'

% location_file = '../data/tweet_locations.txt'
% tag_file = '../data/tweet_words.txt'

location_file = strcat(input_dir, 'tweet_locations.txt')
tag_file = strcat(input_dir, 'tweet_words.txt')

prior_file = strcat(output_dir, 'priors.txt');
mu_file = strcat(output_dir, 'mu.txt');
sigma_file = strcat(output_dir, 'sigma.txt');
pwz_file = strcat(output_dir, 'pwz.txt');
pzr_file = strcat(output_dir, 'pzr.txt');

t = cputime;

locations = importdata(location_file);
raw_tags = importdata(tag_file);
tags = spconvert(raw_tags);

[priors, mu, sigma, Pwz, Pzr] = LGTA(tags, locations, n_region, n_topic, lambdaB);

dlmwrite(prior_file, priors')
dlmwrite(mu_file, mu')
dlmwrite(sigma_file, permute(sigma, [3,1,2]))
dlmwrite(pwz_file, Pwz')
dlmwrite(pzr_file, Pzr')

e = cputime-t;

end
