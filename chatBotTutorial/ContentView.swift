//
//  ContentView.swift
//  chatBotTutorial
//
//  Created by 이명진 on 2023/04/29.
//

import SwiftUI


struct ContentView: View {
    
    @State private var messageText = ""
    @State var messages: [String] = ["안녕하세요 궁금한 문장을 입력하세요."]
    
    
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
            
            ScrollView (.vertical, showsIndicators: false, content: { // scrollbar 삭제
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
                                .font(.system(size:16))
                            
                        }
                        
                    } else { // model
                        
                        HStack{
                            Text("🙂")
                                .font(.title2)
                                .padding(.bottom, 1)
                                .padding(.horizontal, 5)
                                .cornerRadius(5)
                                
                            Spacer()
                        }
                        
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
            }).rotationEffect(.degrees(180))
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
        let url = URL(string: "http://10.90.6.227:8000/predict")!
        
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
                        let botResponse = result["result"] as? String ?? "캐스팅 실패"
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


