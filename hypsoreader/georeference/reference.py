def coordinate_correction(point_file, projection_metadata, originalLat, originalLon):
    # Use Utility File to Extract M
    M = 2

    finalLat = originalLat.copy()
    finalLon = originalLon.copy()

    for i in range(originalLat.shape[0]):
        for j in range(originalLat.shape[1]):
            X = originalLon[i, j]
            Y = originalLat[i, j]
            # Affine Matrix
            # current_coord = np.array([[X], [Y], [1]])
            # res_mult = np.matmul(M, current_coord)
            # newLon = res_mult[0]
            # newLat = res_mult[1]

            # estimateAffinePartial2D

            # current_coord = np.array([[X], [Y], [1]])
            # res_mult = np.matmul(M, current_coord)
            # newLon = res_mult[0]
            # newLat = res_mult[1]

            # Second Degree Polynomial (Scikit)
            # lon_coeff = M.params[0]
            # lat_coeff = M.params[1]
            # newLat, newLon = reference_correction.calculate_poly_geo_coords_skimage(X, Y, lon_coeff, lat_coeff)

            # Np lin alg
            LonM = M[0]
            modifiedLon = LonM[0] * X + LonM[1]

            LatM = M[1]
            modifiedLat = LatM[0] * Y + LatM[1]

            finalLat[i, j] = modifiedLat
            finalLon[i, j] = modifiedLon

    return finalLat, finalLon
