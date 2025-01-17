//
//  ViewController.swift
//  test02
//
//  Created by Anuj Dutt on 7/9/20.
//  Copyright © 2020 Anuj Dutt. All rights reserved.
//

import UIKit
import CoreML
import Accelerate

class ViewController: UIViewController {

    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var yHatPickerView: UIPickerView!
    
    @IBOutlet weak var inputDataShape: UITextField!
    @IBOutlet weak var outputDataShape: UITextField!
    @IBOutlet weak var meanSquaredError: UITextField!
    @IBOutlet weak var estimatedTime: UITextField!
    
    // MARK: Create an Instance of CoreML Model
    //private var model: test02_model_16bit = test02_model_16bit()
    var model = test02_model_16bit()
    
    // Variables to Store Input Data Parameters
    var inputData: [[[[Float32]]]] = []
    var inputShape: [Int] = []
    
    // y_hat .npy file loaded as JSON
    let yHatData = ["0_data", "1_data", "2_data", "3_data", "4_data", "5_data", "6_data", "7_data", "8_data", "9_data", "10_data", "11_data", "12_data", "13_data", "14_data", "15_data", "16_data", "17_data", "18_data", "19_data", "20_data", "21_data", "22_data", "23_data"]
    
    // MSE Calculated for each CoreML Model prior to deployment
    let outputMSE = [26.3428992198896, 23.2058202847838, 59.7725062674726, 156.706916465191, 26.4360885514179, 39.4887757938704, 23.9681476989062, 28.5922287483117, 127.504231547936, 25.9103093543672, 66.2214797688648, 84.7862586291739, 39.8742164979921, 62.1697177004534, 42.0398493573884, 54.6226233112975, 43.4275418709148, 25.6839615860371, 34.6464127112994, 91.2003537872806, 39.2131895700003, 83.3759245258988, 48.3213121871813, 27.4347938597202]
    
    var selectedInputData = ""
    
    @Published var imageData = [ImageLatentSpace]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // MARK: Create an Instance of CoreML Model
        // Prefer NPU if available
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try! test02_model_16bit(configuration: config)
        
        // Input Data Picker View
        yHatPickerView.delegate = self
        yHatPickerView.dataSource = self
    }
    
    // MARK: Function to Make Prediction
    func makePrediction(latentSpaceData: [[[[Float32]]]]) -> UIImage {
        // Create the MLMultiArray for Model Input
        let yHat = try! MLMultiArray(shape: [1, 12, 512, 768], dataType: MLMultiArrayDataType.float32)
        self.inputDataShape.text = "\(yHat.shape)"
        
        // MARK: Populate Data from JSON to MLMultiArray
        for i in 0..<12{
            for j in 0..<512{
                for k in 0..<768{
                    yHat[[0 as NSNumber, i as NSNumber, j as NSNumber, k as NSNumber]] = (latentSpaceData[0][i][j][k]) as NSNumber
                }
            }
        }
        
        // MARK: Make Model Prediction
        let output = try! self.model.prediction(y_hat: yHat)
        let predictions = output.pred
        self.outputDataShape.text = "\(predictions.shape)"
        //print("Model Output: \(predictions)")
        
        // MARK: MLMultiArray to UIIMage Conversion
        //let outputImage = createUIImage(fromFloatArray: predictions)
        return createUIImage(fromFloatArray: predictions)!
    }
    
    
    //MARK: Helper Function to Conver MLMultiArray to UIImage
    func createUIImage(fromFloatArray features: MLMultiArray,
                              min: Float = -1,
                              max: Float = 1) -> UIImage? {
      let cgImg = createCGImage(fromFloatArray: features, min: min, max: max)
      return cgImg.map { UIImage(cgImage: $0) }
    }
    
    
    func createCGImage(fromFloatArray features: MLMultiArray,
                              min: Float = -1,
                              max: Float = 1) -> CGImage? {
      assert(features.dataType == .float32)
      assert(features.shape.count == 4)

      let ptr = UnsafeMutablePointer<Float>(OpaquePointer(features.dataPointer))
        
        // (1, 3, 512, 768)
        let height = features.shape[2].intValue
        let width = features.shape[3].intValue
        let channelStride = features.strides[1].intValue
        let rowStride = features.strides[2].intValue
        let srcRowBytes = rowStride * MemoryLayout<Float>.stride

      var blueBuffer = vImage_Buffer(data: ptr,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: srcRowBytes)
      var greenBuffer = vImage_Buffer(data: ptr.advanced(by: channelStride),
                                      height: vImagePixelCount(height),
                                      width: vImagePixelCount(width),
                                      rowBytes: srcRowBytes)
      var redBuffer = vImage_Buffer(data: ptr.advanced(by: channelStride * 2),
                                    height: vImagePixelCount(height),
                                    width: vImagePixelCount(width),
                                    rowBytes: srcRowBytes)

      let destRowBytes = width * 4
      var pixels = [UInt8](repeating: 0, count: height * destRowBytes)
      var destBuffer = vImage_Buffer(data: &pixels,
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: destRowBytes)
        
        // Assign image as RGB Pixels not BGR
        let error = vImageConvert_PlanarFToBGRX8888(&redBuffer,
                                                    &greenBuffer,
                                                    &blueBuffer,
                                                    Pixel_8(255),
                                                    &destBuffer,
                                                    [max, max, max],
                                                    [min, min, min],
                                                    vImage_Flags(0))
        
      if error == kvImageNoError {
        return CGImage.fromByteArrayRGBA(pixels, width: width, height: height)
      } else {
        return nil
      }
    }
    
    // MARK: Helper Function to increase Image Contrast
    func increaseContrast(_ image: UIImage) -> UIImage {
        let inputImage = CIImage(image: image)!
        let parameters = [
            "inputContrast": NSNumber(value: 2)
        ]
        let outputImage = inputImage.applyingFilter("CIColorControls", parameters: parameters)

        let context = CIContext(options: nil)
        let img = context.createCGImage(outputImage, from: outputImage.extent)!
        return UIImage(cgImage: img)
    }
}


// MARK: Input Data Picker View
extension ViewController: UIPickerViewDataSource {
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return yHatData.count
    }
}

extension ViewController: UIPickerViewDelegate {
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        return yHatData[row]
    }
    
    // MARK: Loading JSON data and Making Prediction
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        print("Data Selected: \(yHatData[row])")
        self.selectedInputData = yHatData[row]
        
        DispatchQueue.main.async {
            self.yHatPickerView.isUserInteractionEnabled = false
            // Make Model Predictions and show Result
            // MARK: Load JSON data from file
            //let data = DataLoader(fileName: self.selectedInputData).imageData
            self.loadJSON(inputFileName: self.selectedInputData) { (result) in
                if (result == true){
                    // Get Data and it's shape from JSON
                   for param in self.imageData{
                       self.inputData = param.data
                       self.inputShape = param.shape
                   }
                    
                    let start = Date()
                    let outputImage = self.makePrediction(latentSpaceData: self.inputData)
                    let end = Date()
                    let executionTime = end.timeIntervalSince(start)
                    let imagesPerSecond = Double(1) / executionTime
                    self.estimatedTime.text = "\(Double(round(imagesPerSecond * 1000) / 1000)) seconds"
                    
                    // Increase Image Contrast
                    let betterContrastImage = self.increaseContrast(outputImage)
                    self.imageView.image = betterContrastImage
                    //let imageView = UIImageView(image: betterContrastImage)
                    //imageView.frame = CGRect(x: 0, y: 0, width: 450, height: 450)
                    //self.view.addSubview(imageView)
                    self.meanSquaredError.text = "\(self.outputMSE[row])"
                    self.yHatPickerView.isUserInteractionEnabled = true
                }
                else{
                    print("Data not loaded...")
                }
            }
        }
    }
    
    
    // MARK: Function to Load Data from JSON file
    func loadJSON(inputFileName: String, completion: (Bool)->()) {
        if let filePath = Bundle.main.url(forResource: inputFileName, withExtension: "json"){
            do {
                let data = try Data(contentsOf: filePath)
                let jsonDecoder = JSONDecoder()
                let dataFromJson = try jsonDecoder.decode([ImageLatentSpace].self, from: data)
                self.imageData = dataFromJson
                completion(true)
            } catch {
              print("Error Encountered: \(error)")
                completion(false)
            }
        }
    }
    
}



// MARK: Helper Extensions
extension CGImage {
  /**
    Helper Function to create a new CGImage from an array of RGBA bytes.
  */
  @nonobjc public class func fromByteArrayRGBA(_ bytes: [UInt8],
                                               width: Int,
                                               height: Int) -> CGImage? {
    return fromByteArray(bytes, width: width, height: height,
                         bytesPerRow: width * 4,
                         colorSpace: CGColorSpaceCreateDeviceRGB(),
                         alphaInfo: .premultipliedLast)
  }

  /**
    Helper Function to create a new CGImage from an array of grayscale bytes.
  */
  @nonobjc public class func fromByteArrayGray(_ bytes: [UInt8],
                                               width: Int,
                                               height: Int) -> CGImage? {
    return fromByteArray(bytes, width: width, height: height,
                         bytesPerRow: width,
                         colorSpace: CGColorSpaceCreateDeviceGray(),
                         alphaInfo: .none)
  }

  @nonobjc class func fromByteArray(_ bytes: [UInt8],
                                    width: Int,
                                    height: Int,
                                    bytesPerRow: Int,
                                    colorSpace: CGColorSpace,
                                    alphaInfo: CGImageAlphaInfo) -> CGImage? {
    return bytes.withUnsafeBytes { ptr in
      let context = CGContext(data: UnsafeMutableRawPointer(mutating: ptr.baseAddress!),
                              width: width,
                              height: height,
                              bitsPerComponent: 8,
                              bytesPerRow: bytesPerRow,
                              space: colorSpace,
                              bitmapInfo: alphaInfo.rawValue)
      return context?.makeImage()
    }
  }
}
