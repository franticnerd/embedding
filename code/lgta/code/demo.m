location_file = '../data/car_image_location.txt'
tag_file = '../data/car_image_tags.txt'

n_region = 50;
n_topic = 10;
lambdaB = 0.1;

locations = importdata(location_file)
raw_tags = importdata(tag_file)
tags = spconvert(raw_tags)

[priors, mu, sigma, Pwz, Pzr] = LGTA(tags, locations, n_region, n_topic, lambdaB)

prior_file = '../data/priors.txt'
mu_file = '../data/mu.txt'
sigma_file = '../data/sigma.txt'
pwz_file = '../data/pwz.txt'
pzr_file = '../data/pzr.txt'
dlmwrite(prior_file, priors)
dlmwrite(mu_file, mu)
dlmwrite(sigma_file, sigma)
dlmwrite(pwz_file, Pwz)
dlmwrite(pzr_file, Pzr)

