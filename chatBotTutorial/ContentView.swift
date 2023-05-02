//
//  ContentView.swift
//  chatBotTutorial
//
//  Created by 이명진 on 2023/04/29.
//

import SwiftUI


struct ContentView: View {
    
    @State private var messageText = ""
    @State var messages: [String] = ["Welcome to App"]
    
    
    var body: some View {
        
        VStack {
            
            HStack{
                Text("Chat")
                    .font(.largeTitle)
                    .bold()
                
                Image(systemName: "bubble.left.fill")
                    .font(.system(size: 26))
                    .foregroundColor(Color.blue)
                
            }
            
            ScrollView {
                ForEach(messages, id: \.self) { message in
                    if message.contains("[USER]") {
                        let newMessage = message.replacingOccurrences(of: "[USER]", with: "")
                        
                        HStack { // user
                            Spacer()
                            Text(newMessage)
                                .padding()
                                .foregroundColor(.white)
                                .background(.blue.opacity(0.8))
                                .cornerRadius(10)
                                .padding(.horizontal, 16)
                                .padding(.bottom, 10)
                            
                        }
                        
                    } else { // model
                        
                        HStack {
                            Text(message)
                                .padding()
                                .background(.gray.opacity(0.15))
                                .cornerRadius(10)
                                .padding(.horizontal, 16)
                                .padding(.bottom, 10)
                            Spacer()
                        }
                        
                    }
                }.rotationEffect(.degrees(180))
            }.rotationEffect(.degrees(180))
                .background(Color.gray.opacity(0.10))
            
            HStack {
                
                TextField("문장을 입력하세요", text: $messageText)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                    .onSubmit {
                        sendMessage(message: messageText)
                    }
                
                Button {
                    sendMessage(message: messageText)
                } label: {
                    Image(systemName: "paperplane.fill")
                }
                .font(.system(size: 26))
                .padding(.horizontal, 10)
                
            }
            .padding()
            

        }
        .padding()
        
    }
    
    
    private func predict(query: String) {
        let url = URL(string: "http://192.168.176.183:8000/predict")!
        // 재혁 192.168.148.196
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        let inputData = ["query": query]  // 입력 데이터
        request.httpBody = try? JSONSerialization.data(withJSONObject: inputData, options: [])

        URLSession.shared.dataTask(with: request) { (data, response, error) in
            if let error = error {
                print("Error:", error)
                return
            }

            guard let data = data else {
                print("No data received")
                return
            }

            do {
                let result = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                if let result = result {
                    DispatchQueue.main.async {
                        let botResponse = result["bot_response"] as? String ?? "Sorry, I didn't understand"
                        self.messages.append(botResponse)
                    }
                }
            } catch let error {
                print("Error serializing JSON:", error)
            }
        }.resume()
    }
    
    
    func sendMessage(message: String) {
        withAnimation {
            messages.append("[USER]" + message)
            // print(messages)
            self.messageText = ""
        }
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            withAnimation {
                predict(query: message)
            }
        }
    }
    
    
    
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}


