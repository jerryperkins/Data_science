var app = new Vue({
    el : '#app',
    data: function() {
        return {
            product: "Americas Leaderboard",
            leaders: [],
            check: false
        }
    },
    mounted: function() {
        // this.testing()
        this.getLeaderboard()
    },

    methods: {
        testing: function() {
            var that = this
            console.log("did this work")
            $.ajax({
                method: "GET",
                url: "https://americas.api.riotgames.com/lor/ranked/v1/leaderboards?api_key=RGAPI-dab42f7e-1209-4aac-9ea9-c8a25a699d01",

            
                success: function(data) {
                    
                    console.log(data.players)
                    for(let i = 0; i < 5    ; i++){
                        that.leaders.push(data.players[i])
                        var player_name = data.players[i].name
                        var puuid = ""
                        $.ajax({
                            method: "GET",
                            url: `https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/${player_name}`,
                            data: {
                                api_key: "RGAPI-dab42f7e-1209-4aac-9ea9-c8a25a699d01"
                            },
                            success: function(data) {
                                console.log("here is summoner data", data)
                                puuid = data.puuid
                                console.log("here is the current player", that.leaders[i])
                                that.leaders[i].puuid = puuid
                                console.log("here is the current player post puuid", that.leaders[i])
                            }
                            // error: function(data, status, error){
                            //     console.log("here is the error data", data)
                            //     $.ajax({
                            //         method: "GET",
                            //         url: `https://br1.api.riotgames.com/lol/summoner/v4/summoners/by-name/${player_name}`,
                            //         data: {
                            //             api_key: "RGAPI-dab42f7e-1209-4aac-9ea9-c8a25a699d01"
                            //         },
                            //         success: function(data,) {
                            //             console.log("here is summoner data", data)
        
        
                            //         },
                            //         error: function(data, status, error){
                            //             console.log("here is the error in brazil call", status)
                            //         }
                            //     })
                            // }
                        })
                        if (that.leaders[i].puuid){
                            console.log("this player has a puuid")
                        }
                    }
                    // that.leaders = data.players
                    console.log("Here are the `${team}`leaders", that.leaders)
                    
                }
            })
        },

        getPlayer: function () {
            console.log("Here are the leaders", this.leaders)
        },

        getLeaderboard: function() {
            var that = this
            axios
            .get("https://americas.api.riotgames.com/lor/ranked/v1/leaderboards?api_key=RGAPI-dab42f7e-1209-4aac-9ea9-c8a25a699d01")
            .then(response => {
                console.log("Here is axios response", response)
                var players = response.data.players
                for(let i = 0; i < 2; i++){
                    that.leaders.push(players[i])
                    // var player_name = data.players[i].name
                    // var puuid = ""
                    axios
                    .get(`https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/${players[i].name}?api_key=RGAPI-dab42f7e-1209-4aac-9ea9-c8a25a699d01`)
                    .then(function (response){
                        console.log("response from na1", response)
                    })
                    .catch(function (response){
                        console.log("na1 error", response.status)
                        console.log("playes name in catch", players[i].name)
                        axios
                        .get(`https://br1.api.riotgames.com/lol/summoner/v4/summoners/by-name/${players[i].name}?api_key=RGAPI-dab42f7e-1209-4aac-9ea9-c8a25a699d01`)
                        .then(function (response){
                            console.log("here is br1 response", response)

                        })
                        .catch(function (response){
                            console.log("error response from br1", response)
                        })
                    })
                }
            })
            .catch(error => {
                console.log(error)
                // this.errored = true
            })
        }
    }
})