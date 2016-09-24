function [priors, mu, sigma, Pwz, Pzr] = LGTA(image_tags, image_locations, n_region, n_topic, lambdaB, topic_prior, topicdist_prior)
% Input:
% image_tags : image-tag matrix (observed data)
%              image_tags(i,j) stores number of occurrences of tag j in image i
% image_locations : image-location matrix (observed data)
%                   image_locations(i, :) stores the lat/log of image i
%
% n_region : # of regions
% n_topic : # of topics
%
% lambdaB : weight for background model
%
% topic_prior (optional) : the prior distribution for topics
% topicdist_prior (optional) : the prior word distributions for topics
%
% Output:
% priors : importance weights of regions
% mu : means of regions
% sigma : covariance matrix of regions
% Pwz : word distributions for topics
% Pzr : topic distributions for regions

[priors0, mu0, sigma0] = EM_init_kmeans(image_locations', n_region);

if ~exist('lambdaB', 'var')
    lambdaB = 0;
end

TopicPriorMode = true;
if ~exist('topic_prior', 'var')
    TopicPriorMode = false;
end

loglik_threshold = 1e-5;
loglik_old = -realmax;
iter = 0;

mu = mu0;
sigma = sigma0;
priors = priors0;

n_region = size(sigma0, 3);
[n_doc, n_word] = size(image_tags);
nnz = size(find(image_tags), 1);

fprintf('region count: %d\n', n_region);
fprintf('doc count: %d\n', n_doc);
fprintf('word count: %d\n', n_word);
fprintf('nonzero count: %d\n', nnz);

Pki = zeros(n_region, n_doc);

Pwz = zeros(n_word, n_topic);
Pzr = zeros(n_topic, n_region);

%Build background model
PwB = sum(image_tags, 1)';
PwB = PwB / sum(PwB);

maxiter = 10;

while iter <= maxiter
    %E-step
    for k = 1:n_region
        p_location = gaussPDF(image_locations', mu(:,k), sigma(:,:,k));
        p_word = ones(n_doc, 1);   
        if iter > 0            
            p_word = image_tags * log((1-lambdaB)*Pwz*Pzr(:,k) + lambdaB*PwB);
            p_word = exp(p_word);
        end
        Pki(k, :) = p_location .* p_word;
    end
    
    if iter > 0
        %Compute the log likelihood
        F = Pki' * priors';
        F(find(F<realmin)) = realmin;
        loglik = mean(log(F));
        fprintf('iteration %d  loglikelihood=%f\n', iter, loglik);
        %Stop the process depending on the increase of the log likelihood 
        if abs((loglik/loglik_old)-1) < loglik_threshold
            break;
        end

        loglik_old = loglik;
    end
    
    iter = iter + 1;
    
    %Compute posterior probability p(i|k)
    Pik_tmp = repmat(priors,n_doc, 1).*Pki';
    Pik = Pik_tmp ./ repmat(sum(Pik_tmp,2), 1, n_region);
    %Compute cumulated posterior probability
    E = sum(Pik);
    
    %M-step
    for k = 1:n_region
        %Update the priors
        priors(k) = E(k) / n_doc;
        %Update the centers
        mu(:,k) = image_locations' * Pik(:,k) / E(k);
        %Update the covariance matrices
        data_tmp1 = image_locations' - repmat(mu(:,k), 1, n_doc);
        sigma(:,:,k) = (repmat(Pik(:,k)', 2, 1) .* data_tmp1*data_tmp1') / E(k);
        %Add a tiny variance to avoid numerical instability
        sigma(:,:,k) = sigma(:,:,k) + 1E-5.*diag(ones(2,1));
    end
    %Update the topics
    if TopicPriorMode
        [Pwz, Pzr] = PLSAPrior(image_tags'*Pik, n_topic, 1e-3, lambdaB, topic_prior, topicdist_prior);
    else
        [Pwz, Pzr] = PLSA(image_tags'*Pik, n_topic, 1e-3, lambdaB);
    end
    
end

