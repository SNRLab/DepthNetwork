#!/bin/bash

# Downloads sample data included with paper into the correct data directory

BASE="$(dirname "$(readlink -f "$0")")"/../..

urls=('https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACuX7w3T3nwVLBs5WvrWLfca/Simple%20phantom/Data/0_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAy32SjMkv08WE0R55p_0SFa/Simple%20phantom/Data/0_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABEkl-B706zs-Lt3oBif26Za/Simple%20phantom/Data/0_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACaMYWPnjh6YeCK_2dgbqwza/Simple%20phantom/Data/2_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAC8I1QYt10auLxfalJ0ltYWa/Simple%20phantom/Data/2_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAU8VKcGqdHAXYqm0HlhBLEa/Simple%20phantom/Data/2_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABUMcEpWnXM0JlY78E75Bywa/Simple%20phantom/Data/3_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABADNlBgUbEsoPFpFkKYmfZa/Simple%20phantom/Data/3_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACgPlwetQEd5pGF9pu6bfhPa/Simple%20phantom/Data/3_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAACcKz_au17ODHzH89NdQrSa/Simple%20phantom/Data/4_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABi-1x5XDbLPR0CdeyBdEcIa/Simple%20phantom/Data/4_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABTwAkM3OMAMiBYw-TdrwqPa/Simple%20phantom/Data/4_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACyZTf5UywBu-Jj212A5Wy5a/Simple%20phantom/Data/5_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAXh3m0mUPdfjoNfSenrStga/Simple%20phantom/Data/5_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACpcOuBKVy1d4ljcJECHur2a/Simple%20phantom/Data/5_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACt320rYBaIapRfTzQojOKYa/Simple%20phantom/Data/6_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAA5yC2weU3h0fxKOqHY7_rGa/Simple%20phantom/Data/6_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAR6Jdwif37Z1JH3CaaCW01a/Simple%20phantom/Data/6_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAnuNjpDjP37Ap-biUfTHAAa/Simple%20phantom/Data/7_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAD2VQnq1qhk2ZvGJ8AHlJ5wa/Simple%20phantom/Data/7_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABm0GIKFik66hd5B1lHsOmqa/Simple%20phantom/Data/7_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAPZR4DjgHHfCVhJlhQ9vcXa/Simple%20phantom/Data/8_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAZmLKbEces2C2YKBQpwwbYa/Simple%20phantom/Data/8_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAB6XZVZ7yYHCQGkqXf2va7Va/Simple%20phantom/Data/8_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAoC29djVtRC_9PfMNAxzAya/Simple%20phantom/Data/9_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACkU4i2e0cpMCqELMGHAACoa/Simple%20phantom/Data/9_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACO1T2wtspxOd885G6sxXL5a/Simple%20phantom/Data/9_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAD_9w4wqwRMkXJ-Xq3wUZFka/Simple%20phantom/Data/10_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAA552Gdhh90qi3pY4NsxvRwa/Simple%20phantom/Data/10_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAkNyit6ClXJHIZRQABJlj5a/Simple%20phantom/Data/10_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACAs4vjbi6MXncpLyT50yxCa/Simple%20phantom/Data/11_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABfuYGyon1ewQElLRH9nTsDa/Simple%20phantom/Data/11_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAAmHTppaY3gZF97P2ogyyuia/Simple%20phantom/Data/11_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABuXmFm4tJykLfTNKfJ0LnCa/Simple%20phantom/Data/12_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AADY41CMDu1_EBLQeZ5QXee0a/Simple%20phantom/Data/12_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAB4hASJ70FSdtJhE5dY6YKfa/Simple%20phantom/Data/12_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABGPP74lWBdZMidKQfIsPNca/Simple%20phantom/Data/13_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AADk6J7z3_3lqTYeOtqybp_ba/Simple%20phantom/Data/13_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AADjtkkwRaegYczWWFxp1zhYa/Simple%20phantom/Data/13_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAB48Mh3X9nO9V_Qd1r5PPmfa/Simple%20phantom/Data/14_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABgqHgwJISZcMiNlzPEiNGoa/Simple%20phantom/Data/14_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AACLXUbewLFE8-iowlygxsufa/Simple%20phantom/Data/14_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAB7wPSxIBrTsWJsGanalEjqa/Simple%20phantom/Data/15_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AADmghDKDb8QYanlr9SFGA0pa/Simple%20phantom/Data/15_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABg2I2d9AFsOW62R20pxFM1a/Simple%20phantom/Data/15_z_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AABgBu8kUleTXbe_qnAF3TRra/Simple%20phantom/Data/16_brdf_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAD5fVDNuGgEzVwtduH3cV6wa/Simple%20phantom/Data/16_rgb_small.hdf5'
      'https://www.dropbox.com/sh/hsn4h4bobss9dnb/AAA00mZTIPetgcjyftenAT2fa/Simple%20phantom/Data/16_z_small.hdf5')

mkdir -p "${BASE}/data/paper"
for url in "${urls[@]}"; do
    file=$(grep -Po '\d{1,2}_(?:z|rgb|brdf)_small\.hdf5' <<< "${url}")
    wget -c "${url}" -O "${BASE}/data/paper/${file}"
done