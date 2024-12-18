# pigeon settings

pigeons_setting = Pigeons.Inputs(target = define_target(),
    n_rounds = 20,
    n_chains = 20,
    multithreaded = true,
    record = [traces; round_trip; record_default()],
    checkpoint=true)