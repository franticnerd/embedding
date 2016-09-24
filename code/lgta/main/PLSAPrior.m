function [Pw_z, Pz_d] = PLSAPrior(X, T, epsilon, lambdaB, topic_prior, topicdist_prior)
%
%
% learns plsa model with T topics from words x docs counts data with prior
% 
% X : (W x D) term-document matrix (observed data)
%     X(i,j) stores number of occurrences of word i in document j
% T       (scalar) : desired number of topics 
% epsilon (scalar) : EM iterations terminate if the relative change in
% log-likelihood is under epsilon
%
% Pw_z (W x T)  : word-topic distributions
% Pz_d (T x D)  : topic-document distributions
%
% lambdaB: weight for background model
% 
% topic_prior : the prior distribution for topics
% topicdist_prior : the prior word distributions for topics

[W,D] = size(X);

%fprintf('--> PLSA started with %d topics, %d documents, and %d word vocabulary.\n', T, D, W);

Pz_d = normalize(rand(T, D), 1);
Pw_z = normalize(rand(W, T), 1);

done = 0;
iter = 0;
max_iter = 100;

P_wB = sum(X, 2);
P_wB = P_wB / sum(P_wB);

old_Pz_d = Pz_d;

while ~done
    iter = iter+ 1;
   
    [w_inds, d_inds] = find(X);  
    nz = size(find(w_inds), 1);

    numerator = zeros(nz, T);
    for i = 1:size(w_inds)
        d = d_inds(i);
        w = w_inds(i);
        for j = 1:T
            numerator(i, j) = Pz_d(j, d) * Pw_z(w, j);
        end
    end
    
    numerator = (1 - lambdaB) * numerator;
    denominator = lambdaB * P_wB(w_inds) + sum(numerator, 2);
    
    Pz_dw = cell(T, 1);
    for i = 1:T
        Pz_dw{i} = sparse(d_inds, w_inds, numerator(:, i) ./ denominator);
    end
    
    for i = 1:T
        Pz_d(i,:) = sum(X' .* Pz_dw{i}, 2)';
    end
    Pz_d = normalize(Pz_d, 1);
    
    for i = 1:T
        if i <= size(topic_prior, 1)
            Pw_z(:,i) = sum(X .* Pz_dw{i}', 2) + topic_prior(i) * topicdist_prior(:, i);
        else
            Pw_z(:,i) = sum(X .* Pz_dw{i}', 2);
        end
    end
    Pw_z = normalize(Pw_z, 1);
    
    if iter > 1
        rel_ch = sum(sum(abs(Pz_d - old_Pz_d))) / (T * D);
        old_Pz_d = Pz_d;
        %fprintf('iteration %3d rel-ch = %.6f \n', iter, rel_ch);
        if iter >= max_iter || (rel_ch < epsilon && (iter > 5))
            done=1;
        end
    end
end