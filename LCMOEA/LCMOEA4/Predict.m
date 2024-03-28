function p = Predict(PopObj)
    pSet = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0];
    for i = 1:length(pSet)
        s(i) = norm(sum(PopObj.^pSet(i),2)-1.0,1);
    end
    p = pSet(find(s==min(s)));
end