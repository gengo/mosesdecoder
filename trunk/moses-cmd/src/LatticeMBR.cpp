/*
 *  LatticeMBR.cpp
 *  moses-cmd
 *
 *  Created by Abhishek Arun on 26/01/2010.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "LatticeMBR.h"
#include "StaticData.h"
#include <algorithm>
#include <set>

size_t bleu_order = 4;
float UNKNGRAMLOGPROB = -20;
void GetOutputWords(const TrellisPath &path, vector <Word> &translation){
	const std::vector<const Hypothesis *> &edges = path.GetEdges();
	
	// print the surface factor of the translation
	for (int currEdge = (int)edges.size() - 1 ; currEdge >= 0 ; currEdge--)
	{
		const Hypothesis &edge = *edges[currEdge];
		const Phrase &phrase = edge.GetCurrTargetPhrase();
		size_t size = phrase.GetSize();
		for (size_t pos = 0 ; pos < size ; pos++)
		{
			translation.push_back(phrase.GetWord(pos));
		}
	}
}


void extract_ngrams(const vector<Word >& sentence, map < Phrase, int >  & allngrams)
{
  for (int k = 0; k < (int)bleu_order; k++)
  {
    for(int i =0; i < max((int)sentence.size()-k,0); i++)
    {
      Phrase ngram(Output);
      for ( int j = i; j<= i+k; j++)
      {
        ngram.AddWord(sentence[j]);
      }
      ++allngrams[ngram];
    }
  }
}



void NgramScores::addScore(const Hypothesis* node, const Phrase& ngram, float score) {
    set<Phrase>::const_iterator ngramIter = m_ngrams.find(ngram);
    if (ngramIter == m_ngrams.end()) {
        ngramIter = m_ngrams.insert(ngram).first;
    }
    map<const Phrase*,float>& ngramScores = m_scores[node];
    map<const Phrase*,float>::iterator scoreIter = ngramScores.find(&(*ngramIter));
    if (scoreIter == ngramScores.end()) {
        ngramScores[&(*ngramIter)] = score;
    } else {
        ngramScores[&(*ngramIter)] = log_sum(score,scoreIter->second);
    }
}

NgramScores::NodeScoreIterator NgramScores::nodeBegin(const Hypothesis* node) {
    return m_scores[node].begin();
}


NgramScores::NodeScoreIterator NgramScores::nodeEnd(const Hypothesis* node)  {
    return m_scores[node].end();
}


void pruneLatticeFB(Lattice & connectedHyp, map < const Hypothesis*, set <const Hypothesis* > > & outgoingHyps, map<const Hypothesis*, vector<Edge> >& incomingEdges, 
                    const vector< float> & estimatedScores, size_t edgeDensity) {
  
  //Need hyp 0 in connectedHyp - Find empty hypothesis
  const Hypothesis* emptyHyp = connectedHyp.at(0);
  while (emptyHyp->GetId() != 0) {
    emptyHyp = emptyHyp->GetPrevHypo();
  }
  connectedHyp.push_back(emptyHyp); //Add it to list of hyps
  
  //Need hyp 0's outgoing Hyps
  for (size_t i = 0; i < connectedHyp.size(); ++i) {
    if (connectedHyp[i]->GetId() > 0 && connectedHyp[i]->GetPrevHypo()->GetId() == 0)
      outgoingHyps[emptyHyp].insert(connectedHyp[i]);
  }
  
  //sort hyps based on estimated scores - do so by copying to multimap
  multimap<float, const Hypothesis*> sortHypsByVal;
  for (size_t i =0; i < estimatedScores.size(); ++i) {
    sortHypsByVal.insert(make_pair<float, const Hypothesis*>(estimatedScores[i], connectedHyp[i]));
  }
  
  multimap<float, const Hypothesis*>::const_iterator it = --sortHypsByVal.end();
  float bestScore = it->first;
  //store best score as score of hyp 0
  sortHypsByVal.insert(make_pair<float, const Hypothesis*>(bestScore, emptyHyp));
  
  
  IFVERBOSE(3) {
    for (multimap<float, const Hypothesis*>::const_iterator it = --sortHypsByVal.end(); it != --sortHypsByVal.begin(); --it) {
      const Hypothesis* currHyp =  it->second;
      cerr << "Hyp " << currHyp->GetId() << ", estimated score: " << it->first << endl;
    }  
  }
  
  
  set <const Hypothesis*> survivingHyps; //store hyps that make the cut in this
  
  size_t numEdgesTotal = edgeDensity * connectedHyp[0]->GetWordsBitmap().GetSize();
  size_t numEdgesCreated = 0;

  float prevScore = -999999;
  
  //now iterate over multimap
  for (multimap<float, const Hypothesis*>::const_iterator it = --sortHypsByVal.end(); it != --sortHypsByVal.begin(); --it) {
    float currEstimatedScore = it->first;
    const Hypothesis* currHyp =  it->second;

    if (numEdgesCreated >= numEdgesTotal && prevScore > currEstimatedScore) //if this hyp has equal estimated score to previous, include its edges too
      break;
    
    prevScore = currEstimatedScore;
    VERBOSE(3, "Num edges created : "<< numEdgesCreated << ", numEdges wanted " << numEdgesTotal << endl)
    VERBOSE(3, "Considering hyp " << currHyp->GetId() << ", estimated score: " << it->first << endl)
    
    survivingHyps.insert(currHyp); //CurrHyp made the cut
    
    // is its best predecessor already included ?
    if (survivingHyps.find(currHyp->GetPrevHypo()) != survivingHyps.end()) { //yes, then add an edge
      vector <Edge>& edges = incomingEdges[currHyp];
      Edge winningEdge(currHyp->GetPrevHypo(),currHyp,currHyp->GetScore() - currHyp->GetPrevHypo()->GetScore(),currHyp->GetTargetPhrase());
      edges.push_back(winningEdge);
      ++numEdgesCreated;
    }
    
    //let's try the arcs too
    const ArcList *arcList = currHyp->GetArcList();
    if (arcList != NULL) {
      ArcList::const_iterator iterArcList;
      for (iterArcList = arcList->begin() ; iterArcList != arcList->end() ; ++iterArcList) {
        const Hypothesis *loserHypo = *iterArcList;
        const Hypothesis* loserPrevHypo = loserHypo->GetPrevHypo();
        if (survivingHyps.find(loserPrevHypo) != survivingHyps.end()) { //found it, add edge
          double arcScore = loserHypo->GetScore() - loserPrevHypo->GetScore(); 
          Edge losingEdge(loserPrevHypo, currHyp, arcScore, loserHypo->GetTargetPhrase());
          vector <Edge>& edges = incomingEdges[currHyp];
          edges.push_back(losingEdge);  
          ++numEdgesCreated;
        }
      }
    }

    //Now if a successor node has already been visited, add an edge connecting the two
    map < const Hypothesis*, set < const Hypothesis* > >::const_iterator outgoingIt = outgoingHyps.find(currHyp);
    
    if (outgoingIt != outgoingHyps.end()) {//currHyp does have successors
      const set<const Hypothesis*> & outHyps = outgoingIt->second; //the successors
      for (set<const Hypothesis*>::const_iterator outHypIts = outHyps.begin(); outHypIts != outHyps.end(); ++outHypIts) {
        const Hypothesis* succHyp = *outHypIts;
        
        if (survivingHyps.find(succHyp) == survivingHyps.end()) //Have we encountered the successor yet?
          continue; //No, move on to next
        
        //Curr Hyp can be : a) the best predecessor  of succ b) or an arc attached to succ
        if (succHyp->GetPrevHypo() == currHyp) { //best predecessor
          vector <Edge>& succEdges = incomingEdges[succHyp];
          Edge succWinningEdge(currHyp, succHyp, succHyp->GetScore() - currHyp->GetScore(), succHyp->GetTargetPhrase());
          succEdges.push_back(succWinningEdge);
          survivingHyps.insert(succHyp);
          ++numEdgesCreated;
        }
        
        //now, let's find an arc
        const ArcList *arcList = succHyp->GetArcList();
        if (arcList != NULL) {
          ArcList::const_iterator iterArcList;
          for (iterArcList = arcList->begin() ; iterArcList != arcList->end() ; ++iterArcList) {
            const Hypothesis *loserHypo = *iterArcList;
            const Hypothesis* loserPrevHypo = loserHypo->GetPrevHypo();
            if (loserPrevHypo == currHyp) { //found it
              vector <Edge>& succEdges = incomingEdges[succHyp];
              double arcScore = loserHypo->GetScore() - currHyp->GetScore(); 
              Edge losingEdge(currHyp, succHyp, arcScore, loserHypo->GetTargetPhrase());
              succEdges.push_back(losingEdge);  
              ++numEdgesCreated;
            }
          }
        }
      }
    }
  }
  
  connectedHyp.clear();
  for (set <const Hypothesis*>::iterator it =  survivingHyps.begin(); it != survivingHyps.end(); ++it) {
    connectedHyp.push_back(*it);
  }
  
  VERBOSE(3, "Done! Num edges created : "<< numEdgesCreated << ", numEdges wanted " << numEdgesTotal << endl)
  
  IFVERBOSE(3) {
    cerr << "Surviving hyps: " ;
    for (set <const Hypothesis*>::iterator it =  survivingHyps.begin(); it != survivingHyps.end(); ++it) {
      cerr << (*it)->GetId() << " ";
    }
    cerr << endl;
  }
}
    
/*vector<Word>  calcMBRSol(Lattice & connectedHyp, map<Phrase, float>& finalNgramScores, const vector<float> & thetas, float p, float r) {
  vector<Word> bestHyp;
  return bestHyp;
}*/


void calcNgramPosteriors(Lattice & connectedHyp, map<const Hypothesis*, vector<Edge> >& incomingEdges, float scale, map<Phrase, float>& finalNgramScores) {
  
  sort(connectedHyp.begin(),connectedHyp.end(),ascendingCoverageCmp); //sort by increasing source word cov
  
  map<const Hypothesis*, float> forwardScore;
  forwardScore[connectedHyp[0]] = 0.0f; //forward score of hyp 0 is 1 (or 0 in logprob space)
  set< const Hypothesis *> finalHyps; //store completed hyps
  
  NgramScores ngramScores;//ngram scores for each hyp 
  
  for (size_t i = 1; i < connectedHyp.size(); ++i) {
    const Hypothesis* currHyp = connectedHyp[i];
    if (currHyp->GetWordsBitmap().IsComplete()) {
      finalHyps.insert(currHyp);
    }
    
    VERBOSE(3, "Processing hyp: " << currHyp->GetId() << ", num words cov= " << currHyp->GetWordsBitmap().GetNumWordsCovered() <<  endl)
    
    vector <Edge> & edges = incomingEdges[currHyp];
    for (size_t e = 0; e < edges.size(); ++e) {
      const Edge& edge = edges[e];
      if (forwardScore.find(currHyp) == forwardScore.end()) {
        forwardScore[currHyp] = forwardScore[edge.GetTailNode()] + edge.GetScore();
        VERBOSE(3, "Fwd score["<<currHyp->GetId()<<"] = fwdScore["<<edge.GetTailNode()->GetId() << "] + edge Score: " << edge.GetScore() << endl)
      }
      else {
        forwardScore[currHyp] = log_sum(forwardScore[currHyp], forwardScore[edge.GetTailNode()] + edge.GetScore());
        VERBOSE(3, "Fwd score["<<currHyp->GetId()<<"] += fwdScore["<<edge.GetTailNode()->GetId() << "] + edge Score: " << edge.GetScore() << endl)
      }
    }
    
    //Process ngrams now
    for (size_t j =0 ; j < edges.size(); ++j) {
      Edge& edge = edges[j];
      const NgramHistory & incomingPhrases = edge.GetNgrams(incomingEdges);
      
      //let's first score ngrams introduced by this edge
      for (NgramHistory::const_iterator it = incomingPhrases.begin(); it != incomingPhrases.end(); ++it) { 
        const Phrase& ngram = it->first;
        const PathCounts& pathCounts = it->second;
        VERBOSE(4, "Calculating score for: " << it->first << endl)
        
        for (PathCounts::const_iterator pathCountIt = pathCounts.begin(); pathCountIt != pathCounts.end(); ++pathCountIt) {
          //Score of an n-gram is forward score of head node of leftmost edge + all edge scores
          const Path&  path = pathCountIt->first;
          float score = forwardScore[path[0]->GetTailNode()]; 
          for (size_t i = 0; i < path.size(); ++i) {
            score += path[i]->GetScore();
          }
          ngramScores.addScore(currHyp,ngram,score);
        }
      }
      
      //Now score ngrams that are just being propagated from the history
      for (NgramScores::NodeScoreIterator it = ngramScores.nodeBegin(edge.GetTailNode()); 
           it != ngramScores.nodeEnd(edge.GetTailNode()); ++it) {
        const Phrase & currNgram = *(it->first);
        float currNgramScore = it->second;
        VERBOSE(4, "Calculating score for: " << currNgram << endl)
        
        if (incomingPhrases.find(currNgram) == incomingPhrases.end()) {
          float score = edge.GetScore() + currNgramScore;
          ngramScores.addScore(currHyp,currNgram,score);
        }
      }
      
    }
  }
  
  float Z = 9999999; //the total score of the lattice
  
  //Done - Print out ngram posteriors for final hyps
  for (set< const Hypothesis *>::iterator finalHyp = finalHyps.begin(); finalHyp != finalHyps.end(); ++finalHyp) {
    const Hypothesis* hyp = *finalHyp;
    
    for (NgramScores::NodeScoreIterator it = ngramScores.nodeBegin(hyp); it != ngramScores.nodeEnd(hyp); ++it) {
        const Phrase& ngram = *(it->first);
        if (finalNgramScores.find(ngram) == finalNgramScores.end()) {
          finalNgramScores[ngram] = it->second;
      }
      else {
          finalNgramScores[ngram] = log_sum(it->second,  finalNgramScores[ngram]);
      }
    }
    
    if (Z == 9999999) {
      Z = forwardScore[hyp];
    }
    else {
      Z = log_sum(Z, forwardScore[hyp]);
    }
  }
  
  Z *= scale;  //scale the score
  
  for (map<Phrase, float>::iterator finalScoresIt = finalNgramScores.begin();  finalScoresIt != finalNgramScores.end(); ++finalScoresIt) {
    finalScoresIt->second =  finalScoresIt->second * scale - Z;
    IFVERBOSE(2) {
      cout << finalScoresIt->first << " [" << finalScoresIt->second << "]" << endl;
    }
  }
  
}

const NgramHistory& Edge::GetNgrams(map<const Hypothesis*, vector<Edge> > & incomingEdges)  {
  
  if (m_ngrams.size() > 0)
    return m_ngrams;
  
  const Phrase& currPhrase = GetWords();
  //Extract the n-grams local to this edge
  for (size_t start = 0; start < currPhrase.GetSize(); ++start) {
    for (size_t end = start; end < start + bleu_order; ++end) {
      if (end < currPhrase.GetSize()){
        Phrase edgeNgram(Output);
        for (size_t index = start; index <= end; ++index) {
          edgeNgram.AddWord(currPhrase.GetWord(index));  
        }
        //cout << "Inserting Phrase : " << edgeNgram << endl;
        vector<const Edge*> edgeHistory;
        edgeHistory.push_back(this);
        storeNgramHistory(edgeNgram, edgeHistory);
      }
      else {
        break;
      }
    }
  }
  
  map<const Hypothesis*, vector<Edge> >::iterator it = incomingEdges.find(m_tailNode);
  if (it != incomingEdges.end()) { //node has incoming edges
    vector<Edge> & inEdges = it->second;
    
    for (vector<Edge>::iterator edge = inEdges.begin(); edge != inEdges.end(); ++edge) {//add the ngrams straddling prev and curr edge
      const NgramHistory & edgeIncomingNgrams = edge->GetNgrams(incomingEdges);
      for (NgramHistory::const_iterator edgeInNgramHist = edgeIncomingNgrams.begin(); edgeInNgramHist != edgeIncomingNgrams.end(); ++edgeInNgramHist) {
        const Phrase& edgeIncomingNgram = edgeInNgramHist->first;
        const PathCounts &  edgeIncomingNgramPaths = edgeInNgramHist->second;
        size_t back = min(edgeIncomingNgram.GetSize(), edge->GetWordsSize());
        const Phrase&  edgeWords = edge->GetWords();
        IFVERBOSE(3) {
          cerr << "Edge: "<< *edge <<endl;
          cerr << "edgeWords: " << edgeWords << endl;
          cerr << "edgeInNgram: " << edgeIncomingNgram << endl;  
        }
        
        Phrase edgeSuffix(Output);
        Phrase ngramSuffix(Output);
        GetPhraseSuffix(edgeWords,back,edgeSuffix);
        GetPhraseSuffix(edgeIncomingNgram,back,ngramSuffix);
        
        if (ngramSuffix == edgeSuffix) { //we've got the suffix of previous edge
          size_t  edgeInNgramSize =  edgeIncomingNgram.GetSize();
          
          for (size_t i = 0; i < GetWordsSize() && i + edgeInNgramSize < bleu_order ; ++i){
            Phrase newNgram(edgeIncomingNgram);
            for (size_t j = 0; j <= i ; ++j){
              newNgram.AddWord(GetWords().GetWord(j));
            }
            VERBOSE(3, "Inserting New Phrase : " << newNgram << endl)
            
            for (PathCounts::const_iterator pathIt = edgeIncomingNgramPaths.begin(); pathIt !=  edgeIncomingNgramPaths.end(); ++pathIt) {
              Path newNgramPath = pathIt->first;
              newNgramPath.push_back(this);
              storeNgramHistory(newNgram, newNgramPath, pathIt->second);  
            }
          }
        }
      }
    }  
  }
  return m_ngrams;
}

//Add the last lastN words of origPhrase to targetPhrase
void Edge::GetPhraseSuffix(const Phrase&  origPhrase, size_t lastN, Phrase& targetPhrase) const {
  size_t origSize = origPhrase.GetSize();
  size_t startIndex = origSize - lastN;
  for (size_t index = startIndex; index < origPhrase.GetSize(); ++index) {
    targetPhrase.AddWord(origPhrase.GetWord(index));
  }
}

bool Edge::operator< (const Edge& compare ) const {
  if (m_headNode->GetId() < compare.m_headNode->GetId())
    return true;
  if (compare.m_headNode->GetId() < m_headNode->GetId())
    return false;
  if (m_tailNode->GetId() < compare.m_tailNode->GetId())
    return true;
  if (compare.m_tailNode->GetId() < m_tailNode->GetId())
    return false;
  return GetScore() <  compare.GetScore();
}

ostream& operator<< (ostream& out, const Edge& edge) {
  out << "Head: " << edge.m_headNode->GetId() << ", Tail: " << edge.m_tailNode->GetId() << ", Score: " << edge.m_score << ", Phrase: " << edge.m_targetPhrase << endl;
  return out;
}
  
bool ascendingCoverageCmp(const Hypothesis* a, const Hypothesis* b) {
  return a->GetWordsBitmap().GetNumWordsCovered() <  b->GetWordsBitmap().GetNumWordsCovered();
}

vector<Word>  calcMBRSol(const TrellisPathList& nBestList, map<Phrase, float>& finalNgramScores, const vector<float> & thetas, float p, float r){
  
  vector<float> mbrThetas = thetas;
  if (thetas.size() == 0) { //thetas not specified on the command line, use p and r instead
    mbrThetas.push_back(-1); //Theta 0
    mbrThetas.push_back(1/(bleu_order*p));
    for (size_t i = 2; i <= bleu_order; ++i){
      mbrThetas.push_back(mbrThetas[i-1] / r);  
    }
  }
  IFVERBOSE(2) {  
  cout << "Thetas: ";
  for (size_t i = 0; i < mbrThetas.size(); ++i) {
    cout << mbrThetas[i] << " ";
  }
  cout << endl;
  }
  
  float argmaxScore = -1e20;
  TrellisPathList::const_iterator iter;
  size_t ctr = 0;
  
  vector<Word> argmaxTranslation;
  for (iter = nBestList.begin() ; iter != nBestList.end() ; ++iter, ++ctr)
  {
    const TrellisPath &path = **iter;
    // get words in translation
    vector<Word> translation;
    GetOutputWords(path, translation);
    
    // collect n-gram counts
    map < Phrase, int > counts;
    extract_ngrams(translation,counts);
    
    //Now score this translation
    float mbrScore = mbrThetas[0] * translation.size();
    
    float ngramScore = 0;
    
    for (map < Phrase, int >::iterator ngrams = counts.begin(); ngrams != counts.end(); ++ngrams) {
      float ngramPosterior = UNKNGRAMLOGPROB; 
      map<Phrase,float>::const_iterator ngramPosteriorIt = finalNgramScores.find(ngrams->first);
      if (ngramPosteriorIt != finalNgramScores.end()) {
        ngramPosterior = ngramPosteriorIt->second;
      }
          
      if (ngramScore == 0) {
        ngramScore = log((double) ngrams->second) + ngramPosterior + log(mbrThetas[(ngrams->first).GetSize()]);
      }
      else {
        ngramScore = log_sum(ngramScore, float(log((double) ngrams->second) + ngramPosterior + log(mbrThetas[(ngrams->first).GetSize()])));
      }
      //cout << "Ngram: " << ngrams->first << endl;
    } 

    mbrScore += exp(ngramScore);

    if (mbrScore > argmaxScore){
      argmaxScore = mbrScore;
      IFVERBOSE(2) {
        cout << "HYP " << ctr << " IS NEW BEST: ";
        for (size_t i = 0; i < translation.size(); ++i)
          cout << translation[i]  ;
        cout << "[" << argmaxScore << "]" << endl;    
      }
      argmaxTranslation = translation;
    }
  }
  return argmaxTranslation;
}

vector<Word> doLatticeMBR(Manager& manager, TrellisPathList& nBestList) {
    const StaticData& staticData = StaticData::Instance();
    std::map < int, bool > connected;
    std::vector< const Hypothesis *> connectedList;
    map<Phrase, float> ngramPosteriors;
    std::map < const Hypothesis*, set <const Hypothesis*> > outgoingHyps;
    map<const Hypothesis*, vector<Edge> > incomingEdges;
    vector< float> estimatedScores;
    manager.GetForwardBackwardSearchGraph(&connected, &connectedList, &outgoingHyps, &estimatedScores);
    pruneLatticeFB(connectedList, outgoingHyps, incomingEdges, estimatedScores, staticData.GetLatticeMBRPruningFactor());
    calcNgramPosteriors(connectedList, incomingEdges, staticData.GetMBRScale(), ngramPosteriors);      
    vector<Word> mbrBestHypo = calcMBRSol(nBestList, ngramPosteriors, staticData.GetLatticeMBRThetas(), 
            staticData.GetLatticeMBRPrecision(), staticData.GetLatticeMBRPRatio());
    return mbrBestHypo;
}
