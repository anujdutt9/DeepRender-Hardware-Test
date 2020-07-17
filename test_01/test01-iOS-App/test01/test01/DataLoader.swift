//
//  DataLoader.swift
//  test01
//
//  Created by Anuj Dutt on 7/7/20.
//  Copyright © 2020 Anuj Dutt. All rights reserved.
//

import Foundation

// MARK: Class to load data (y_hat) from JSON file
public class DataLoader {
    
    @Published var imageData = [ImageLatentSpace]()
    
    // Set the FileName of JSON File to Read
    let fileName: String = "15_data"
    
    // Initializer
    init() {
        self.loadJSON()
    }
    
    // MARK: Function to Load Data from JSON file
    func loadJSON() {
        if let filePath = Bundle.main.url(forResource: fileName, withExtension: "json"){
            do {
                let data = try Data(contentsOf: filePath)
                let jsonDecoder = JSONDecoder()
                let dataFromJson = try jsonDecoder.decode([ImageLatentSpace].self, from: data)
                self.imageData = dataFromJson
            } catch {
              print("Error Encountered: \(error)")
            }
        }
        
    }
}
