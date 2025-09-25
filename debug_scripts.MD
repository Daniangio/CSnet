i=0;j=1;e = data['edge_index'];l = data['edge_lengths'];print(torch.sort(l[e[0]==i]).values);print(torch.sort(l[e[0]==j]).values)

i=0;j=1;e = data['edge_index'];l = data['edge_lengths'];a = torch.sort(l[e[0]==i]).indices;b = torch.sort(l[e[0]==j]).indices;x = data['edge_lengths'][e[0]==i];y = data['edge_lengths'][e[0]==j];q = x[a];w = y[b];print((q-w).abs().max())

i=0;j=1;e = data['edge_index'];l = data['edge_lengths'];a = torch.sort(l[e[0]==i]).indices;b = torch.sort(l[e[0]==j]).indices;x = latents[e[0]==i];y = latents[e[0]==j];q = x[a];w = y[b];print((q-w).abs().max())

i=0;j=1;e = data['edge_index'];l = data['edge_lengths'];a = torch.sort(l[e[0]==i]).indices;b = torch.sort(l[e[0]==j]).indices;x = data['edge_radial_attrs'][e[0]==i];y = data['edge_radial_attrs'][e[0]==j];q = x[a];w = y[b];print((q-w).abs().max());print(q[:4]);print(w[:4])