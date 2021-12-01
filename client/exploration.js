var app = new Vue({
    el: '#app',
    data: {
        checkedDimension: null,
        curiosityWeight: null,
        dataset: '',
        dimensions: [],
        distance: null,
        explorationRunning: false,
        foundItemsWithRatio: {},
        getPredictedScores: false,
        getscores: true,
        greedySum: false,
        guidanceAlgorithm: "RLSum",
        guidanceMode: "Partially guided",
        guidanceWeightMode: "decreasing",
        galaxy_class_scores: null,
        history: [],
        inputSet: null,
        loadedCuriosityWeight: null,
        loadedTargetSet: null,
        loading: false,
        loadSteps: [],
        minSetSize: 200,
        minUniformity: 2,
        novelty: null,
        operationStates: [],
        operator: 'by_facet',
        orderedDimensions: {},
        predictedDimension: "",
        predictedOperation: "",
        predictedScores: [],
        predictedSetId: null,
        previousOperations: [],
        progress: -1,
        resultSetCount: 10,
        seenPredicates: [],
        selectedSetId: null,
        sets: [],
        seenSets: [],
        setStates: [],
        stepCounter: 0,
        swapExecuted: false,
        targetItems: [],
        targetSet: null,
        totalReward: 0,
        uniformity: null,
        utility: null,
        utilityWeights: [0.00005, 0.00005, 0.9999],
        webService: ''
    },
    mounted() {
        this.loading = true
        let query_params = new URLSearchParams(window.location.search.substring(1));
        this.dataset = query_params.get("dataset")
        this.greedySum = query_params.get("greedysum") == 'true'
        this.webService = window.location.origin
        let url = new URL(this.webService + "/app/" + "get-dataset-information")
        url.searchParams.append("dataset", this.dataset)
        axios.get(url).then(response => {
            this.dimensions = response.data.dimensions.map((x) => x.replace(this.dataset + '.', ''))
            this.orderedDimensions = {}
            _.forIn(response.data.ordered_dimensions, (value, key) => {
                this.orderedDimensions[key.replace(this.dataset + '.', '')] = value
            })
        })
        // url = new URL(this.webService + "/app/" + "get-target-items-and-prediction")
        // url.searchParams.append("target_set", this.targetSet)
        // url.searchParams.append("curiosity_weight", this.curiosityWeight)
        // url.searchParams.append("dataset_ids", [])
        // axios.get(url).then(response => {
        //     this.targetItems = response.data.targetItems
        //     this.foundItemsWithRatio = response.data.foundItemsWithRatio
        //     this.predictedOperation = response.data.predictedOperation
        //     this.operator = this.predictedOperation
        //     this.predictedDimension = response.data.predictedDimension
        //     this.checkedDimension = this.predictedDimension
        //     this.predictedSetId = response.data.predictedSetId
        //     this.selectedSetId = this.predictedSetId
        //     this.setStates = response.data.setStates
        //     this.operationStates = response.data.operationStates
        //     this.loadedCuriosityWeight = this.curiosityWeight
        //     this.loadedTargetSet = this.targetSet
        //     this.loading = false
        // })
        this.loading = false
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl)
        })
    },
    computed: {
        facetDimensions: function () {
            if (this.selectedSetId == null) {
                return this.dimensions
            } else {
                return _.difference(this.dimensions, this.sets.find((x) => x.id === this.selectedSetId).predicate.map((x) => x.dimension.replace(this.dataset + '.', '')))
            }
        },
        neighborsDimensions: function () {
            if (this.selectedSetId == null) {
                return Object.keys(this.orderedDimensions)
            } else {
                return _.intersection(Object.keys(this.orderedDimensions), this.sets.find((x) => x.id === this.selectedSetId).predicate.map((x) => x.dimension.replace(this.dataset + '.', '')))
            }
        },

        saveLink: function () {
            return 'data:attachment/json,' + encodeURI(JSON.stringify(this.history))
        },
        selectedSetPredicateCount: function () {
            if (this.selectedSetId) {
                return this.sets.find((x) => x.id === this.selectedSetId).predicate.length
            }
            return 0
        },
        isLoading: function () {
            if (this.loadSteps.length == 0) {
                return false
                // } else if (this.inputSet != this.loadSteps[0].inputSet |
                //     this.selectedSetId != this.loadSteps[0].selectedSetId |
                //     this.operator != this.loadSteps[0].operator |
                //     this.checkedDimension != this.loadSteps[0].checkedDimension) {
                //     this.loadSteps = []
                //     this.history = []
                //     return false
            } else {
                return true
            }
        }
    },
    methods: {
        url: function (item) {
            return `http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra=${item.ra}&dec=${item.dec}&scale=0.4&width=120&height=120`
        },
        set_id: function (set) {
            return set.id < 0 ? "No Id " + Math.abs(set.id) : set.id
        },
        loadModel() {
            this.loading = true
            this.explorationRunning = false
            requestData = {
                get_scores: this.getscores,
                get_predicted_scores: this.getPredictedScores,
                seen_predicates: this.seenPredicates,
                seen_sets: this.seenSets,
                dataset_to_explore: this.dataset,
                utility_weights: this.utilityWeights
            }

            requestData.dataset_ids = this.sets.map(a => a.id)
            requestData.weights_mode = this.guidanceWeightMode
            url = new URL(this.webService + "/app/" + "load-model")
            axios.put(url, requestData).then(response => {
                this.foundItemsWithRatio = response.data.foundItemsWithRatio
                this.predictedOperation = response.data.predictedOperation
                this.operator = this.predictedOperation
                this.predictedDimension = response.data.predictedDimension
                this.checkedDimension = this.predictedDimension
                this.predictedSetId = response.data.predictedSetId
                this.selectedSetId = this.predictedSetId
                this.setStates = response.data.setStates
                this.operationStates = response.data.operationStates
                this.loadedCuriosityWeight = this.curiosityWeight
                this.loadedTargetSet = this.targetSet
                setId = this.selectedSetId
                setTimeout(function () {
                    if (setId) {
                        document.getElementById("set-" + setId).scrollIntoView()
                    }
                }, 0)
                this.loading = false
            })

        },
        getSwapSummary() {
            this.loading = true
            const url = new URL(this.webService + "/swapSum")
            url.searchParams.append("dataset_to_explore", this.dataset)
            url.searchParams.append("min_set_size", this.minSetSize)
            url.searchParams.append("min_uniformity_target", this.minUniformity)
            url.searchParams.append("result_set_count", this.resultSetCount)
            axios.get(url).then(response => {
                response.data.sets.forEach(set => set.predicate.forEach(item => item.dimension = item.dimension.replace(this.dataset + '.', '')))
                this.sets = this.sortSets(response.data.sets)
                this.utility = response.data.utility
                this.uniformity = response.data.uniformity
                this.novelty = response.data.novelty
                this.distance = response.data.distance
                this.swapExecuted = true
                this.galaxy_class_scores = response.data.galaxy_class_scores
                this.loading = false
                this.loadModel()
            })
        },
        submitted() {
            if (this.sets.length !== 0 & this.selectedSetId === null) {
                alert("Please click anywhere on a set to select the next input set.")
            } else if ((this.operator === 'by_facet' | this.operator === 'by_neighbors') & !this.checkedDimension) {
                alert("Please select at least one dimension.")
            } else {
                this.loading = true
                this.inputSet = _.find(this.sets, (set) => set.id === this.selectedSetId)
                const url = new URL(this.webService + "/operators/" + this.operator + '-g')

                requestData = {
                    get_scores: this.getscores,
                    get_predicted_scores: this.getPredictedScores,
                    seen_predicates: this.seenPredicates,
                    seen_sets: this.seenSets,
                    dataset_to_explore: this.dataset,
                    utility_weights: this.utilityWeights
                }
                if (!this.inputSet) {
                    requestData.input_set_id = -1
                    this.inputSet = null
                } else {
                    requestData.input_set_id = this.inputSet.id
                }
                if (this.operator === 'by_facet' | this.operator === 'by_neighbors') {
                    requestData.dimensions = [this.dataset + "." + this.checkedDimension]
                }

                // requestData.target_set = this.targetSet
                // requestData.curiosity_weight = this.curiosityWeight
                // requestData.target_items = this.targetItems
                // requestData.found_items_with_ratio = this.foundItemsWithRatio
                requestData.previous_set_states = this.setStates
                requestData.previous_operation_states = this.operationStates
                requestData.target_set = null
                requestData.curiosity_weight = null
                requestData.target_items = null
                requestData.found_items_with_ratio = null
                // requestData.previous_set_states = null
                // requestData.previous_operation_states = null
                requestData.previous_operations = this.previousOperations
                requestData.dataset_ids = this.sets.map(a => a.id)
                requestData.decreasing_gamma = this.guidanceWeightMode == "decreasing"
                requestData.weights_mode = this.guidanceWeightMode
                requestData.galaxy_class_scores = this.galaxy_class_scores
                this.predictedSetId = null
                this.predictedDimension = null
                axios.put(url, requestData).then(response => {
                    if (this.inputSet)
                        delete this.inputSet.data
                    let reward = 0

                    reward = response.data.reward
                    this.totalReward += reward

                    this.history.push({
                        "selectedSetId": this.selectedSetId,
                        "operator": this.operator,
                        "checkedDimension": this.checkedDimension,
                        "url": url,
                        "inputSet": this.inputSet,
                        "reward": reward,
                        "requestData": requestData,
                        "curiosityReward": response.data.curiosityReward,
                        "utility": response.data.utility,
                        "uniformity": response.data.uniformity,
                        "novelty": response.data.novelty,
                        "distance": response.data.distance,
                        "utilityWeights": response.data.utility_weights,
                        "galaxy_class_score": response.data.galaxy_class_score,
                        "class_score_found_12": response.data.class_score_found_12,
                        "class_score_found_15": response.data.class_score_found_15,
                        "class_score_found_18": response.data.class_score_found_18,
                        "class_score_found_21": response.data.class_score_found_21
                    })

                    if (this.history.length > 3) {
                        delete this.history[this.history.length - 4].requestData
                    }
                    this.checkedDimension = null
                    this.selectedSetId = null
                    response.data.sets.forEach(set => set.predicate.forEach(item => item.dimension = item.dimension.replace(this.dataset + '.', '')))
                    this.sets = this.sortSets(response.data.sets)
                    this.utility = response.data.utility
                    this.uniformity = response.data.uniformity
                    this.novelty = response.data.novelty
                    this.distance = response.data.distance
                    this.seenPredicates = response.data.seenPredicates
                    this.predictedScores = response.data.predictedScores
                    if (['decreasing', 'increasing'].indexOf(this.guidanceWeightMode) != -1)
                        this.utilityWeights = response.data.utility_weights
                    this.seenSets = response.data.seen_sets
                    this.previousOperations = response.data.previous_operations
                    this.galaxy_class_scores = response.data.galaxy_class_scores
                    this.loadStep(false)
                    this.loading = false
                    // this.foundItemsWithRatio = response.data.foundItemsWithRatio
                    // if (this.guidanceMode != "Manual") {
                    //     this.predictedOperation = response.data.predictedOperation
                    //     this.operator = this.predictedOperation
                    //     this.predictedDimension = response.data.predictedDimension
                    //     this.checkedDimension = this.predictedDimension
                    //     this.predictedSetId = response.data.predictedSetId
                    //     this.selectedSetId = this.predictedSetId
                    // } else {
                    //     this.predictedOperation = null
                    //     this.predictedDimension = null
                    //     this.predictedSetId = null
                    //     this.selectedSetId = null
                    // }
                    if (this.guidanceMode != 'Manual') {
                        if (this.guidanceAlgorithm == 'Top1Sum') {
                            prediction = this.predictedScores[0]
                            this.loadPrediction(prediction.setId, prediction.operation, prediction.attribute, false)
                            this.predictedOperation = this.operator
                            this.predictedDimension = this.checkedDimension
                        } else {
                            this.predictedOperation = response.data.predictedOperation
                            this.operator = this.predictedOperation
                            this.predictedDimension = response.data.predictedDimension
                            this.checkedDimension = this.predictedDimension
                            this.predictedSetId = response.data.predictedSetId
                            this.selectedSetId = this.predictedSetId
                        }
                    } else if (!this.isLoading()) {
                        this.predictedOperation = null
                        this.predictedDimension = null
                        this.predictedSetId = null
                        this.selectedSetId = null
                    }
                    this.setStates = response.data.setStates
                    this.operationStates = response.data.operationStates
                    this.stepCounter = this.stepCounter + 1
                    setId = this.selectedSetId
                    historyCardIndex = this.history.length - 1
                    setTimeout(function () {
                        document.getElementById("operation-" + historyCardIndex).scrollIntoView()
                        if (setId) {
                            document.getElementById("set-" + setId).scrollIntoView()
                        }
                    }, 0)

                    if (this.explorationRunning) {
                        this.wait(true)
                    }
                })
            }
        },
        sortSets: function (sets) {
            if (this.history.length > 0) {
                lastAction = this.history[this.history.length - 1]
                if (lastAction.checkedDimension) {
                    try {

                        var sortDimension = lastAction.checkedDimension
                        var sortDimensionValues = this.orderedDimensions[sortDimension]
                        if (sortDimensionValues && sortDimensionValues.length > 0) {
                            setComparison = function (set1, set2) {
                                return sortDimensionValues.indexOf(set1.predicate.find(x => x.dimension == sortDimension).value) - sortDimensionValues.indexOf(set2.predicate.find(x => x.dimension == sortDimension).value)
                            }
                            return sets.sort(setComparison)
                        } else {
                            return sets
                        }
                    } catch (err) {
                        console.error(err)
                        return sets
                    }
                }
            }
            return sets
        },
        wait(starting) {
            if (!this.explorationRunning)
                this.progress = -1
            else {
                if (starting) {
                    this.progress = 0
                } else {
                    this.progress += 1
                }
                if (this.progress == 50) {
                    this.submitted()
                } else {
                    instance = this
                    setTimeout(() => {
                        instance.wait(false)
                    }, 80);
                }
            }
        },
        undo() {
            this.loading = true
            this.operator = "undo"
            this.history.pop()
            this.predictedSetId = null
            this.predictedDimension = null
            previous_state = this.history[this.history.length - 1]
            axios.put(previous_state.url, previous_state.requestData).then(response => {
                this.inputSet = previous_state.inputSet
                this.checkedDimension = null
                this.selectedSetId = null
                response.data.sets.forEach(set => set.predicate.forEach(item => item.dimension = item.dimension.replace(this.dataset + '.', '')))
                this.sets = this.sortSets(response.data.sets)
                this.utility = response.data.utility
                this.uniformity = response.data.uniformity
                this.novelty = response.data.novelty
                this.seenPredicates = response.data.seenPredicates
                this.predictedScores = response.data.predictedScores
                if (['decreasing', 'increasing'].indexOf(this.guidanceWeightMode) != -1)
                    this.utilityWeights = response.data.utility_weights
                this.operator = previous_state.operator
                this.foundItemsWithRatio = response.data.foundItemsWithRatio
                this.guidanceModeChange()

                this.setStates = response.data.setStates
                this.operationStates = response.data.operationStates
                this.stepCounter = this.stepCounter - 1
                this.loading = false
            })
        },
        startExploration() {
            this.explorationRunning = true
            this.submitted()
        },
        restart() {
            this.loading = true
            this.operator = 'by_facet'
            this.guidanceMode = 'Partially guided'
            this.explorationRunning = false
            this.inputSet = null
            this.selectedSetId = null
            this.sets = []
            this.history = []
            this.utility = null
            this.uniformity = null
            this.distance = null
            this.predictedScores = []
            this.novelty = null
            this.seenPredicates = []
            this.guidanceAlgorithm = "RLSum"
            this.guidanceMode = "Partially guided"
            this.guidanceWeightMode = "decreasing"
            this.utilityWeights = [0.00005, 0.00005, 0.9999]
            this.predictedSetId = null
            this.predictedDimension = null
            this.previousOperations = []
            this.swapExecuted = false
            this.stepCounter = 0
            // url = new URL(this.webService + "/app/" + "get-target-items-and-prediction")
            // url.searchParams.append("target_set", this.targetSet)
            // url.searchParams.append("curiosity_weight", this.curiosityWeight)
            // url.searchParams.append("dataset_ids", [])
            // axios.get(url).then(response => {
            //     this.targetItems = response.data.targetItems
            //     this.foundItemsWithRatio = response.data.foundItemsWithRatio
            //     this.predictedOperation = response.data.predictedOperation
            //     this.operator = this.predictedOperation
            //     this.predictedDimension = response.data.predictedDimension
            //     this.checkedDimension = this.predictedDimension
            //     this.predictedSetId = response.data.predictedSetId
            //     this.selectedSetId = this.predictedSetId
            //     this.setStates = response.data.setStates
            //     this.operationStates = response.data.operationStates
            //     this.loading = false
            // })
            this.loading = false
        },
        loadPipeline(event) {
            if (event.target.files.length != 0) {
                const reader = new FileReader();
                reader.addEventListener('load', (readEvent) => {
                    console.log(readEvent.target.result)
                    this.loadSteps = JSON.parse(readEvent.target.result)
                    // this.sets = []
                    this.loadStep(true)
                    event.target.value = ""
                })
                reader.readAsText(event.target.files[0])
            }
        },
        loadStep(isFirstStep) {
            if (this.loadSteps.length > 0) {
                if (!isFirstStep) {
                    this.loadSteps.shift()
                }
                if (this.loadSteps.length > 0) {
                    this.inputSet = this.loadSteps[0].inputSet
                    this.selectedSetId = this.loadSteps[0].selectedSetId
                    this.operator = this.loadSteps[0].operator
                    this.checkedDimension = this.loadSteps[0].checkedDimension
                }
            }
        },
        setClass(id) {
            if (this.setClicked === id) {
                return "row selected"
            }
            else {
                return "row"
            }
        },
        setClicked(set) {
            if (set.id >= 0) {
                this.selectedSetId = set.id
                if (this.selectedSetPredicateCount < 2 && (this.operator == "by_superset" || this.operator == "by_distribution")) {
                    this.operator = "by_facet"
                }
            }
        },
        loadPrediction(setId, operation, attribute, scrollIntoView) {
            this.selectedSetId = setId
            this.operator = operation
            this.checkedDimension = attribute
            if (scrollIntoView)
                document.getElementById("set-" + setId).scrollIntoView()
        },
        isSelectedPrediction(setId, operation, attribute) {
            if (operation == "by_facet" || operation == "by_neighbors") {
                return this.selectedSetId == setId && this.operator == operation && this.checkedDimension == attribute
            } else {
                return this.selectedSetId == setId && this.operator == operation
            }
        },
        guidanceModeChange() {
            console.log(this.guidanceMode)
            if (this.guidanceMode != 'Manual') {
                this.guidanceAlgorithmChange()
            } else {
                this.predictedOperation = null
                this.predictedDimension = null
                this.predictedSetId = null
                this.selectedSetId = null
                this.predictedScores = []
                this.getPredictedScores = false
            }
            this.explorationRunning = false
        },
        guidanceAlgorithmChange() {
            if (this.guidanceAlgorithm == 'Top1Sum') {
                this.getPredictedScores = true
                this.loading = true
                const url = new URL(this.webService + "/get_predicted_scores")

                requestData = {
                    get_scores: this.getscores,
                    get_predicted_scores: this.getPredictedScores,
                    seen_predicates: this.seenPredicates,
                    seen_sets: this.seenSets,
                    dataset_to_explore: this.dataset,
                    utility_weights: this.utilityWeights
                }
                if (!this.inputSet) {
                    requestData.input_set_id = -1
                    this.inputSet = null
                } else {
                    requestData.input_set_id = this.inputSet.id
                }

                requestData.previous_operations = this.previousOperations
                requestData.dataset_ids = this.sets.map(a => a.id)
                axios.put(url, requestData).then(response => {
                    response.data.sets.forEach(set => set.predicate.forEach(item => item.dimension = item.dimension.replace(this.dataset + '.', '')))
                    this.sets = this.sortSets(response.data.sets)
                    this.utility = response.data.utility
                    this.uniformity = response.data.uniformity
                    this.novelty = response.data.novelty
                    this.distance = response.data.distance
                    this.seenPredicates = response.data.seenPredicates
                    this.predictedScores = response.data.predictedScores
                    if (['decreasing', 'increasing'].indexOf(this.guidanceWeightMode) != -1)
                        this.utilityWeights = response.data.utility_weights
                    prediction = this.predictedScores[0]
                    this.loadPrediction(prediction.setId, prediction.operation, prediction.attribute, false)
                    this.loading = false
                })
            } else {
                this.getPredictedScores = false
                if (this.guidanceAlgorithm == "RLSum") {
                    this.loadModel()
                }
            }
        },
        guidanceWeightModeChange() {
            if (this.guidanceWeightMode == "increasing") {
                this.utilityWeights = [0.5, 0.5, 0]
            } else if (this.guidanceWeightMode == "decreasing") {
                this.utilityWeights = [0.0005, 0.0005, 0.999]
            } else if (this.guidanceWeightMode == "high") {
                this.utilityWeights = [0.1, 0.1, 0.8]
            } else if (this.guidanceWeightMode == "low") {
                this.utilityWeights = [0.45, 0.45, 0.1]
            } else if (this.guidanceWeightMode == "balanced") {
                this.utilityWeights = [0.333, 0.333, 0.334]
            }
            this.guidanceAlgorithmChange()
        }
    }
})