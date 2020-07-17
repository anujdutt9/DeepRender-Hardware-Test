//
//  ImageLatentSpace.swift
//  test01
//
//  Created by Anuj Dutt on 7/7/20.
//  Copyright © 2020 Anuj Dutt. All rights reserved.
//

import Foundation

// MARK: Struct for Input Data (y_hat)
struct ImageLatentSpace: Codable {
    var data: [[[[Float32]]]]
    var shape: [Int]
}
