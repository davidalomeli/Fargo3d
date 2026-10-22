def read_par(folder):
    params = {}
    #returns .par file info
    with open(f'{folder}variables.par', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    params[parts[0].strip()] = parts[1].strip()
    return params
