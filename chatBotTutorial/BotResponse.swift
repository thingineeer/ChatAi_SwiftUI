//
//  BotResponse.swift
//  chatBotTutorial
//
//  Created by 이명진 on 2023/04/29.
//

import Foundation // 수정 할 부분

func getBotResponse(message: String) -> String {
    let tempMessage = message.lowercased()
    
    if tempMessage.contains("hello"){
        return "Hey there!"
    } else if tempMessage.contains("goodbye"){
        return "Talk to you later!"
    } else if tempMessage.contains("how are you") {
        return "I'm fine, how about you?"
    } else {
        return "That's cool."
    }
}
